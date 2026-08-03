#!/usr/bin/env python3
"""
Build the proto-MRD kernel K(z | z_form) for one population model
and one formation-redshift bin.

For a fixed z_form, this script computes

    K(z | z_form) =
        \int dlogM p(logM | z_form)
        \int dfEdd p(fEdd | logM, z_form)
        N_BH(logM, fEdd)
        \int dt [Y(t | logM, fEdd) / t] delta[z - z_merg(z_form, t)]

This script does NOT multiply by nAGN(z_form).
It reads per-run yield histograms produced by run_yields.py and N_BH from
the per-run summary file.

It can be run directly from CLI, or imported and called from a wrapper.

Additionally, if per-run 2D histograms H(t_delay, Mchirp) are present, the
script builds the redshift-dependent chirp-mass distribution implied by the
same cosmological weights and saves Mchirp percentiles vs merger redshift.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator, interp1d


# ============================================================
# ---------------------- USER SETTINGS ------------------------
# ============================================================

DEFAULT_MCHIRP_PERCENTILES = (5.0, 16.0, 50.0, 84.0, 95.0)


# ============================================================
# ---------------------- I/O HELPERS -------------------------
# ============================================================

def grid_cell_widths_1d(x: np.ndarray) -> np.ndarray:
    """
    Return quadrature cell widths associated with each grid point x[i].

    For interior points:
        dx[i] = 0.5 * (x[i+1] - x[i-1])

    For edges:
        dx[0]  = x[1] - x[0]
        dx[-1] = x[-1] - x[-2]

    Works for nonuniform monotonically increasing grids.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1D array with at least 2 points")

    dx = np.empty_like(x)
    dx[1:-1] = 0.5 * (x[2:] - x[:-2])
    dx[0] = x[1] - x[0]
    dx[-1] = x[-1] - x[-2]

    return dx


def get_nearest_cell_width(x: float, grid: np.ndarray, dgrid: np.ndarray, name: str = "grid") -> float:
    """
    Return the cell width associated with the nearest grid point to x.
    """
    grid = np.asarray(grid, dtype=float)
    dgrid = np.asarray(dgrid, dtype=float)

    if len(grid) != len(dgrid):
        raise ValueError(f"{name}: grid and dgrid must have same length")

    idx = np.argmin(np.abs(grid - x))
    return float(dgrid[idx])


def load_pM_given_z(npzfile: str):
    """
    Expected arrays in npz:
        z_grid      shape (Nz,)
        M_grid      shape (NM,)
        pM_given_z  shape (Nz, NM)

    Returns function p_M_given_z(M, z).
    Assumes M_grid is log10(M/Msun).
    """
    d = np.load(npzfile)
    z_grid = d["z_grid"]
    M_grid = d["M_grid"]
    p_grid = d["pM_given_z"]

    interp = RegularGridInterpolator(
        (z_grid, M_grid),
        p_grid,
        bounds_error=False,
        fill_value=0.0,
    )

    dlogM_grid = grid_cell_widths_1d(M_grid)

    def p_M_given_z(M, z):
        z = np.atleast_1d(z).astype(float)
        M = np.full_like(z, float(M))
        pts = np.column_stack([z, M])
        return interp(pts)

    return p_M_given_z, M_grid, dlogM_grid


def load_pfedd_given_Mz(npzfile: str):
    """
    Expected arrays in npz:
        z_grid            shape (Nz,)
        M_grid            shape (NM,)
        fedd_grid         shape (Nf,)
        pfedd_given_Mz    shape (Nz, NM, Nf)

    Returns
    -------
    p_fedd_given_Mz : callable
        Function p_fedd_given_Mz(fedd, M, z)
    fedd_grid : ndarray
    dfedd_grid : ndarray
    """
    d = np.load(npzfile)
    z_grid = d["z_grid"]
    M_grid = d["M_grid"]
    fedd_grid = d["fedd_grid"]
    p_grid = d["pfedd_given_Mz"]

    interp = RegularGridInterpolator(
        (z_grid, M_grid, fedd_grid),
        p_grid,
        bounds_error=False,
        fill_value=0.0,
    )

    # Keep the current behavior of the original script.
    dfedd_grid = 1.0  # grid_cell_widths_1d(fedd_grid)

    def p_fedd_given_Mz(fedd, M, z):
        z = np.atleast_1d(z).astype(float)
        M = np.full_like(z, float(M))
        fedd = np.full_like(z, float(fedd))
        pts = np.column_stack([z, M, fedd])
        return interp(pts)

    return p_fedd_given_Mz, fedd_grid, dfedd_grid


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+|\t+", engine="python", comment="#")


def read_mchirp_hist2d(npzfile: Path) -> dict:
    """
    Expected arrays in npz:
        H_tdelay_mchirp_given_M_fedd   shape (Nt, Nm)
        t_delay_edges_Gyr              shape (Nt+1,)
        t_delay_centers_Gyr            shape (Nt,)
        mchirp_edges_Msun              shape (Nm+1,)
        mchirp_centers_Msun            shape (Nm,)
    """
    d = np.load(npzfile)
    required = {
        "H_tdelay_mchirp_given_M_fedd",
        "t_delay_edges_Gyr",
        "t_delay_centers_Gyr",
        "mchirp_edges_Msun",
        "mchirp_centers_Msun",
    }
    missing = required.difference(d.files)
    if missing:
        raise KeyError(f"Missing arrays in {npzfile}: {sorted(missing)}")

    return {
        "H": np.asarray(d["H_tdelay_mchirp_given_M_fedd"], dtype=float),
        "t_edges": np.asarray(d["t_delay_edges_Gyr"], dtype=float),
        "t_centers": np.asarray(d["t_delay_centers_Gyr"], dtype=float),
        "m_edges": np.asarray(d["mchirp_edges_Msun"], dtype=float),
        "m_centers": np.asarray(d["mchirp_centers_Msun"], dtype=float),
    }


# ============================================================
# -------------------- COSMOLOGY HELPERS ---------------------
# ============================================================

def build_cosmo_tables(zmax: float = 20.0, nz: int = 200000):
    """
    Build reusable cosmology interpolation tables.

    Returns
    -------
    age_to_z : interp1d
        Maps cosmic age [Gyr] -> redshift
    z_grid : ndarray
        Redshift grid
    age_grid : ndarray
        Cosmic age(z) [Gyr]
    Hz_grid : ndarray
        H(z) [Gyr^-1]
    """
    z_grid = np.linspace(0.0, zmax, nz)
    age_grid = Planck18.age(z_grid).to_value(u.Gyr)
    Hz_grid = Planck18.H(z_grid).to_value(1 / u.Gyr)

    age_to_z = interp1d(
        age_grid[::-1],
        z_grid[::-1],
        bounds_error=False,
        fill_value=(z_grid[-1], z_grid[0]),
    )

    return age_to_z, z_grid, age_grid, Hz_grid


def get_cosmo_cache(zmax: float = 20.0, nz: int = 200000):
    age_to_z, z_grid, age_grid, Hz_grid = build_cosmo_tables(zmax=zmax, nz=nz)
    return {
        "age_to_z": age_to_z,
        "z_grid": z_grid,
        "age_grid": age_grid,
        "Hz_grid": Hz_grid,
    }


def z_from_ageform_and_tdelay(
    age_form: float,
    age_today: float,
    t_delay_gyr: np.ndarray,
    age_to_z,
):
    """
    Convert formation age + delay time into merger redshift.
    Returns NaN for mergers after today or invalid times.
    """
    t_delay_gyr = np.asarray(t_delay_gyr, dtype=float)
    age_merg = age_form + t_delay_gyr

    ok = np.isfinite(age_merg) & (t_delay_gyr > 0.0) & (age_merg <= age_today + 1e-10)

    z_merg = np.full_like(t_delay_gyr, np.nan, dtype=float)
    z_merg[ok] = age_to_z(age_merg[ok])
    return z_merg


def precompute_time_to_redshift_mapping(
    t_grid: np.ndarray,
    age_form: float,
    age_today: float,
    age_to_z,
):
    """
    For a fixed z_form, precompute the mapping from the common t-grid
    to merger redshift, along with approximate dt associated with each bin center.
    """
    t_grid = np.asarray(t_grid, dtype=float)

    ok = np.isfinite(t_grid) & (t_grid > 0.0)
    t_grid = t_grid[ok]

    if len(t_grid) < 2:
        raise ValueError("Need at least 2 valid t-grid points.")

    logt = np.log10(t_grid)
    dlogt = np.diff(logt)

    dlogt_eff = np.empty_like(logt)
    dlogt_eff[1:-1] = 0.5 * (dlogt[:-1] + dlogt[1:])
    dlogt_eff[0] = dlogt[0]
    dlogt_eff[-1] = dlogt[-1]

    dt = t_grid * np.log(10.0) * dlogt_eff

    z_merg = z_from_ageform_and_tdelay(
        age_form=age_form,
        age_today=age_today,
        t_delay_gyr=t_grid,
        age_to_z=age_to_z,
    )

    valid_mask = np.isfinite(z_merg) & np.isfinite(dt) & (t_grid > 0.0)

    return {
        "reference_t_grid": t_grid,
        "valid_mask": valid_mask,
        "t": t_grid[valid_mask],
        "dt": dt[valid_mask],
        "z_merg": z_merg[valid_mask],
    }


# ============================================================
# -------------------- RUN-DISCOVERY HELPERS -----------------
# ============================================================

RUN_RE = re.compile(r"logM_(?P<logM>[-+0-9.]+)/fEdd_(?P<fedd>[-+0-9.eE]+)/")


def parse_run_params_from_path(run_dir: Path):
    m = RUN_RE.search(run_dir.as_posix())
    if m is None:
        raise ValueError(f"Could not parse logM/fEdd from path: {run_dir}")
    return float(m.group("logM")), float(m.group("fedd"))


def find_summary_file(run_dir: Path):
    """
    Prefer the yield-only summary if present.
    """
    candidates = [
        run_dir / "summary_yields.txt",
        run_dir / "summary.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No summary file found in {run_dir}")


# ============================================================
# ------------------ KERNEL CONTRIBUTIONS --------------------
# ============================================================

def build_run_contribution(
    yield_file: Path,
    N_BH: float,
    pM: float,
    dlogM: float,
    pfedd: float,
    dfedd: float,
    z_bins: np.ndarray,
    mapping: dict,
):
    """
    Map a single run yield histogram into a contribution to K(z | z_form),
    using a precomputed common t -> z_merg mapping for this z_form.
    """
    df = read_table(yield_file)

    required = {"t_delay_Gyr", "Y_tdelay_given_M_fedd"}
    if not required.issubset(df.columns):
        raise KeyError(
            f"{yield_file} must contain columns "
            f"'t_delay_Gyr' and 'Y_tdelay_given_M_fedd'"
        )

    t_all = pd.to_numeric(df["t_delay_Gyr"], errors="coerce").to_numpy(dtype=float)
    Y_all = pd.to_numeric(df["Y_tdelay_given_M_fedd"], errors="coerce").to_numpy(dtype=float)

    ref_t = mapping["reference_t_grid"]
    t_ok = np.isfinite(t_all) & (t_all > 0.0)
    t_clean = t_all[t_ok]

    if len(t_clean) != len(ref_t) or not np.allclose(t_clean, ref_t, rtol=0, atol=1e-12):
        raise ValueError(f"Inconsistent t grid in {yield_file}")

    Y_clean = Y_all[t_ok]

    valid_mask = mapping["valid_mask"]
    Y = Y_clean[valid_mask]
    t = mapping["t"]
    dt = mapping["dt"]
    z_merg = mapping["z_merg"]

    ok = np.isfinite(Y) & (Y >= 0.0)
    Y = Y[ok]
    t = t[ok]
    dt = dt[ok]
    z_merg = z_merg[ok]

    weight = N_BH * pM * dlogM * pfedd * dfedd * (Y / t) * dt

    finite = np.isfinite(z_merg) & np.isfinite(weight) & (weight >= 0.0)
    z_merg = z_merg[finite]
    weight = weight[finite]

    hist, _ = np.histogram(z_merg, bins=z_bins, weights=weight)
    dz = np.diff(z_bins)
    K_bin = hist / dz

    return K_bin, {
        "int_Y_over_t_dt": float(np.sum((Y / t) * dt)),
        "valid_bins": int(len(z_merg)),
    }


def build_run_mchirp_contribution(
    hist2d_file: Path,
    N_BH: float,
    pM: float,
    dlogM: float,
    pfedd: float,
    dfedd: float,
    z_bins: np.ndarray,
    mapping: dict,
    m_edges_ref: np.ndarray | None,
    m_centers_ref: np.ndarray | None,
):
    """
    Map a single-run H(t_delay, Mchirp) histogram into a contribution to the
    redshift-dependent chirp-mass distribution.

    Returns
    -------
    mass_by_z : ndarray, shape (Nz, Nm)
        Weighted mass histogram integrated over Mchirp bins.
        Summing over Mchirp gives the same integrated contribution that enters
        the proto-kernel before division by dz.
    diag : dict
    m_edges : ndarray
    m_centers : ndarray
    """
    d = read_mchirp_hist2d(hist2d_file)
    H = d["H"]
    t_centers = d["t_centers"]
    m_edges = d["m_edges"]
    m_centers = d["m_centers"]

    if m_edges_ref is not None and not np.allclose(m_edges, m_edges_ref, rtol=0, atol=1e-12):
        raise ValueError(f"Inconsistent Mchirp bin edges in {hist2d_file}")
    if m_centers_ref is not None and not np.allclose(m_centers, m_centers_ref, rtol=0, atol=1e-12):
        raise ValueError(f"Inconsistent Mchirp bin centers in {hist2d_file}")

    ref_t = mapping["reference_t_grid"]
    t_ok = np.isfinite(t_centers) & (t_centers > 0.0)
    t_clean = t_centers[t_ok]
    if len(t_clean) != len(ref_t) or not np.allclose(t_clean, ref_t, rtol=0, atol=1e-12):
        raise ValueError(f"Inconsistent t grid in {hist2d_file}")

    H_clean = H[t_ok, :]
    valid_mask = mapping["valid_mask"]

    if H_clean.shape[0] != len(valid_mask):
        raise ValueError(f"Unexpected H(t, Mchirp) shape in {hist2d_file}: {H_clean.shape}")

    H_use = H_clean[valid_mask, :]
    t = mapping["t"]
    dt = mapping["dt"]
    z_merg = mapping["z_merg"]

    dm = np.diff(m_edges)
    if np.any(~np.isfinite(dm)) or np.any(dm <= 0.0):
        raise ValueError(f"Invalid Mchirp bin widths in {hist2d_file}")

    row_prefactor = N_BH * pM * dlogM * pfedd * dfedd * (dt / t)
    finite_row = np.isfinite(z_merg) & np.isfinite(row_prefactor) & (row_prefactor >= 0.0)

    z_idx = np.digitize(z_merg[finite_row], z_bins) - 1
    in_range = (z_idx >= 0) & (z_idx < len(z_bins) - 1)
    z_idx = z_idx[in_range]

    H_rows = H_use[finite_row, :][in_range, :]
    row_prefactor = row_prefactor[finite_row][in_range]

    row_masses = H_rows * row_prefactor[:, None] * dm[None, :]

    mass_by_z = np.zeros((len(z_bins) - 1, len(m_centers)), dtype=float)
    for iz, row_mass in zip(z_idx, row_masses):
        mass_by_z[iz] += row_mass

    return mass_by_z, {
        "valid_delay_bins": int(len(z_idx)),
        "integrated_mass_contribution": float(np.sum(mass_by_z)),
    }, m_edges, m_centers


def weighted_percentile_from_binned_hist(
    m_edges: np.ndarray,
    mass_hist: np.ndarray,
    percentiles: np.ndarray,
) -> np.ndarray:
    """
    Compute approximate weighted percentiles from a binned mass histogram,
    assuming uniform density within each Mchirp bin.
    """
    percentiles = np.asarray(percentiles, dtype=float)
    out = np.full(percentiles.shape, np.nan, dtype=float)

    total = float(np.sum(mass_hist))
    if (not np.isfinite(total)) or total <= 0.0:
        return out

    cdf_edges = np.concatenate([[0.0], np.cumsum(mass_hist)]) / total
    targets = percentiles / 100.0

    for i, q in enumerate(targets):
        q = float(np.clip(q, 0.0, 1.0))
        idx = np.searchsorted(cdf_edges, q, side="right") - 1
        idx = int(np.clip(idx, 0, len(m_edges) - 2))

        c0 = cdf_edges[idx]
        c1 = cdf_edges[idx + 1]
        x0 = m_edges[idx]
        x1 = m_edges[idx + 1]

        if c1 <= c0:
            out[i] = 0.5 * (x0 + x1)
        else:
            frac = (q - c0) / (c1 - c0)
            frac = float(np.clip(frac, 0.0, 1.0))
            out[i] = x0 + frac * (x1 - x0)

    return out


# ============================================================
# ------------------------- CORE RUN -------------------------
# ============================================================

def run_proto_mrd(
    runs_base,
    alpha,
    label,
    pm_file,
    pfedd_file,
    redshift_model,
    zform,
    zmax=10.5,
    nz=200,
    outdir="../outputs/protoMRD",
    yield_labels=("1g", "ng"),
    cosmo_cache=None,
    mchirp_percentiles=DEFAULT_MCHIRP_PERCENTILES,
):
    runs_base = Path(runs_base) / f"alpha_{alpha}"
    outdir = Path(outdir) / f"alpha_{alpha}" / label / redshift_model / f"{zform}"
    outdir.mkdir(parents=True, exist_ok=True)

    p_M_given_z, M_grid, dlogM_grid = load_pM_given_z(pm_file)
    p_fedd_given_Mz, fedd_grid, dfedd_grid = load_pfedd_given_Mz(pfedd_file)

    z_bins = np.linspace(0.0, zmax, nz + 1)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

    if cosmo_cache is None:
        cosmo_zmax = max(zmax, zform, 15.0) + 2.0
        cosmo_cache = get_cosmo_cache(zmax=cosmo_zmax)

    age_to_z = cosmo_cache["age_to_z"]
    age_form = Planck18.age(zform).to_value(u.Gyr)
    age_today = Planck18.age(0.0).to_value(u.Gyr)

    run_dirs = list(runs_base.glob(f"logM_*/fEdd_*/{label}/outputs"))
    if len(run_dirs) == 0:
        raise RuntimeError(f"No runs found under {runs_base} with label {label}")

    summary_files = []
    for run_dir in run_dirs:
        try:
            summary_files.append(find_summary_file(run_dir))
        except FileNotFoundError:
            print(f"[warning] skipping run with no summary file: {run_dir}")
            continue

    if len(summary_files) == 0:
        raise FileNotFoundError(
            f"No usable summary files found under {runs_base} for label={label}"
        )

    common_mapping = None
    for summary_file in summary_files:
        run_dir = summary_file.parent
        for yl in yield_labels:
            yfile = run_dir / f"yield_{yl}.txt"
            if yfile.exists():
                try:
                    df_tmp = read_table(yfile)
                    if "t_delay_Gyr" not in df_tmp.columns:
                        continue
                    t_grid = pd.to_numeric(
                        df_tmp["t_delay_Gyr"], errors="coerce"
                    ).to_numpy(dtype=float)

                    common_mapping = precompute_time_to_redshift_mapping(
                        t_grid=t_grid,
                        age_form=age_form,
                        age_today=age_today,
                        age_to_z=age_to_z,
                    )
                    break
                except Exception as e:
                    print(f"[warning] could not use {yfile} as reference t-grid: {e}")
                    continue
        if common_mapping is not None:
            break

    if common_mapping is None:
        raise FileNotFoundError(
            f"Could not find any readable yield_*.txt file for label={label}"
        )

    contribution_rows = []
    K_total = np.zeros_like(z_centers)

    mchirp_mass_by_z_total = None
    mchirp_edges_ref = None
    mchirp_centers_ref = None
    mchirp_rows = []

    for summary_file in summary_files:
        run_dir = summary_file.parent

        try:
            logM, fedd = parse_run_params_from_path(run_dir)
        except Exception as e:
            print(f"[warning] skipping {run_dir}: could not parse logM/fEdd ({e})")
            continue

        try:
            df_summary = read_table(summary_file)
        except Exception as e:
            print(f"[warning] could not read {summary_file}: {e}")
            continue

        if len(df_summary) == 0:
            print(f"[warning] empty summary file: {summary_file}")
            continue

        pM_val = float(np.atleast_1d(p_M_given_z(logM, zform))[0])
        pfedd_val = float(np.atleast_1d(p_fedd_given_Mz(fedd, logM, zform))[0])
        dlogM = get_nearest_cell_width(logM, M_grid, dlogM_grid, name="M_grid")
        dfedd = 1.0  # get_nearest_cell_width(fedd, fedd_grid, dfedd_grid, name="fedd_grid")

        if not np.isfinite(pM_val) or pM_val < 0.0:
            pM_val = 0.0
        if not np.isfinite(pfedd_val) or pfedd_val < 0.0:
            pfedd_val = 0.0
        if not np.isfinite(dlogM) or dlogM <= 0.0:
            raise ValueError(f"Invalid dlogM for logM={logM}")
        if not np.isfinite(dfedd) or dfedd <= 0.0:
            raise ValueError(f"Invalid dfedd for fEdd={fedd}")

        for yl in yield_labels:
            this = df_summary[df_summary["label"] == yl]
            if len(this) == 0:
                continue

            yield_file = run_dir / f"yield_{yl}.txt"
            if not yield_file.exists():
                print(f"[warning] missing {yield_file}, skipping")
                continue

            if "N_BH" not in this.columns:
                print(f"[warning] N_BH column missing in {summary_file}, skipping {yl}")
                continue

            N_BH = float(pd.to_numeric(this["N_BH"], errors="coerce").mean())
            if not np.isfinite(N_BH) or N_BH <= 0.0:
                continue

            try:
                K_bin, diag = build_run_contribution(
                    yield_file=yield_file,
                    N_BH=N_BH,
                    pM=pM_val,
                    pfedd=pfedd_val,
                    dlogM=dlogM,
                    dfedd=dfedd,
                    z_bins=z_bins,
                    mapping=common_mapping,
                )
            except Exception as e:
                print(f"[warning] failed on {yield_file}: {e}")
                continue

            K_total += K_bin

            contribution_rows.append({
                "run_dir": str(run_dir),
                "label": yl,
                "logM": logM,
                "fEdd": fedd,
                "z_form": zform,
                "pM_given_zform": pM_val,
                "pfedd_given_M_zform": pfedd_val,
                "dlogM": dlogM,
                "dfEdd": dfedd,
                "pM_times_dlogM": pM_val * dlogM,
                "pfedd_times_dfEdd": pfedd_val * dfedd,
                "N_BH": N_BH,
                "int_Y_over_t_dt": diag["int_Y_over_t_dt"],
                "valid_delay_bins": diag["valid_bins"],
                "integrated_contribution": float(np.sum(K_bin * np.diff(z_bins))),
            })

            hist2d_file = run_dir / f"hist2d_tdelay_mchirp_{yl}.npz"
            if hist2d_file.exists():
                try:
                    mass_by_z, diag_m, m_edges_ref, m_centers_ref = build_run_mchirp_contribution(
                        hist2d_file=hist2d_file,
                        N_BH=N_BH,
                        pM=pM_val,
                        dlogM=dlogM,
                        pfedd=pfedd_val,
                        dfedd=dfedd,
                        z_bins=z_bins,
                        mapping=common_mapping,
                        m_edges_ref=mchirp_edges_ref,
                        m_centers_ref=mchirp_centers_ref,
                    )
                    if mchirp_mass_by_z_total is None:
                        mchirp_mass_by_z_total = np.zeros_like(mass_by_z)
                    mchirp_mass_by_z_total += mass_by_z
                    mchirp_edges_ref = m_edges_ref
                    mchirp_centers_ref = m_centers_ref

                    mchirp_rows.append({
                        "run_dir": str(run_dir),
                        "label": yl,
                        "logM": logM,
                        "fEdd": fedd,
                        "z_form": zform,
                        "integrated_mchirp_contribution": diag_m["integrated_mass_contribution"],
                        "valid_mchirp_delay_bins": diag_m["valid_delay_bins"],
                        "hist2d_file": str(hist2d_file),
                    })
                except Exception as e:
                    print(f"[warning] failed on {hist2d_file}: {e}")
            else:
                print(f"[warning] missing {hist2d_file}, skipping Mchirp percentiles for this run")

    df_kernel = pd.DataFrame({
        "z": np.round(z_centers, 3),
        "K_proto": K_total,
    })
    df_kernel.to_csv(outdir / "kernel_vs_z.txt", sep="\t", index=False)

    df_contrib = pd.DataFrame(contribution_rows)
    df_contrib.to_csv(outdir / "contributions_by_run.txt", sep="\t", index=False)

    if mchirp_mass_by_z_total is not None and mchirp_edges_ref is not None:
        q_list = np.asarray(mchirp_percentiles, dtype=float)
        rows = []
        for iz, zc in enumerate(z_centers):
            mass_hist = mchirp_mass_by_z_total[iz]
            qvals = weighted_percentile_from_binned_hist(
                m_edges=mchirp_edges_ref,
                mass_hist=mass_hist,
                percentiles=q_list,
            )
            row = {
                "z": np.round(zc, 3),
                "mass_weight_total": float(np.sum(mass_hist)),
            }
            for q, val in zip(q_list, qvals):
                qname = f"mchirp_p{str(q).replace('.', 'p')}_Msun"
                row[qname] = val
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            outdir / "mchirp_percentiles_vs_z.txt",
            sep="\t",
            index=False,
        )

        np.savez_compressed(
            outdir / "mchirp_distribution_vs_z.npz",
            z_edges=z_bins,
            z_centers=z_centers,
            mchirp_edges_Msun=mchirp_edges_ref,
            mchirp_centers_Msun=mchirp_centers_ref,
            mass_by_z=mchirp_mass_by_z_total,
        )

        pd.DataFrame(mchirp_rows).to_csv(
            outdir / "mchirp_contributions_by_run.txt",
            sep="\t",
            index=False,
        )

    meta = pd.DataFrame([{
        "redshift_model": redshift_model,
        "label": label,
        "z_form": zform,
        "zmin": z_bins[0],
        "zmax": z_bins[-1],
        "nz": nz,
        "nruns_used": len(df_contrib),
        "mchirp_percentiles": ",".join(map(str, mchirp_percentiles)),
        "has_mchirp_summary": bool(mchirp_mass_by_z_total is not None),
    }])
    meta.to_csv(outdir / "meta.txt", sep="\t", index=False)

    print(f"Done. Wrote proto-MRD kernel to {outdir}")
    if mchirp_mass_by_z_total is not None:
        print(f"Done. Wrote Mchirp percentiles to {outdir / 'mchirp_percentiles_vs_z.txt'}")


# ============================================================
# -------------------------- CLI -----------------------------
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--runs-base", required=True, help="Base directory containing alpha_*/logM_*/fEdd_*/")
    p.add_argument("--alpha", required=True, help="Alpha value used to select alpha_<alpha>/")
    p.add_argument("--label", required=True,
                   help="Run label between fEdd and outputs")
    p.add_argument("--pm-file", required=True, help="Path to pM_given_z npz")
    p.add_argument("--pfedd-file", required=True, help="Path to pfedd_given_Mz npz")
    p.add_argument("--redshift-model", required=True, help="Label, e.g. SE or EL")
    p.add_argument("--zform", required=True, type=float, help="Formation redshift bin center")
    p.add_argument("--zmax", type=float, default=10.5)
    p.add_argument("--nz", type=int, default=200, help="Number of merger-z bins")
    p.add_argument("--outdir", default="../outputs/protoMRD")
    p.add_argument(
        "--yield-labels",
        nargs="+",
        default=["1g", "ng"],
        help="Which yield files to include",
    )
    p.add_argument(
        "--mchirp-percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_MCHIRP_PERCENTILES),
        help="Percentiles of Mchirp to save vs merger redshift",
    )

    return p.parse_args()


def main():
    args = parse_args()
    run_proto_mrd(
        runs_base=args.runs_base,
        alpha=args.alpha,
        label=args.label,
        pm_file=args.pm_file,
        pfedd_file=args.pfedd_file,
        redshift_model=args.redshift_model,
        zform=args.zform,
        zmax=args.zmax,
        nz=args.nz,
        outdir=args.outdir,
        yield_labels=tuple(args.yield_labels),
        cosmo_cache=None,
        mchirp_percentiles=tuple(args.mchirp_percentiles),
    )


if __name__ == "__main__":
    main()
