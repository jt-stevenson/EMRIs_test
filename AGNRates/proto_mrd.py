#!/usr/bin/env python3
r"""
Build the proto-MRD kernel K(z | z_form) for one population model
and one formation-redshift bin.

For a fixed z_form, this script computes

    K(z | z_form) =
        \int dlogM p(logM | z_form)
        \int dfEdd p(fEdd | logM, z_form)
        \int dlogZ p(logZ | z_form)
        N_BH(logM, fEdd)
        \int dt [Y(t | logM, fEdd, Z) / t] delta[z - z_merg(z_form, t)]

This script does NOT multiply by nAGN(z_form).
It reads per-run yield histograms produced by run_yields.py and N_BH from
the per-run summary file. The expected compact yield layout is

    <yields-base>/Z<Z>/SG_alpha_<alpha>/<label>/
        logM_<logM>_fEdd_<fEdd>/

It can be run directly from CLI, or imported and called from a wrapper.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import ndtr


# ============================================================
# ---------------------- USER SETTINGS ------------------------
# ============================================================

VALID_METALLICITY_MODELS = {"flat_Z", "evolving_Z", "single_Z"}
Z_SUN = 0.02
SIGMA_LOGZ = 0.5
METALLICITY_MODEL_PARAMETERS = {
    "flat_Z": {"a_Z": 1.0, "b_Z": 0.0},
    "evolving_Z": {"a_Z": 1.04, "b_Z": -0.24},
}
GYR_TO_YR = 1.0e9


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
    delay_bin_widths: np.ndarray | None = None,
):
    """
    For a fixed z_form, precompute the mapping from the common t-grid
    to merger redshift, along with approximate dt associated with each bin center.
    """
    # print('computing redshift from time')
    t_grid = np.asarray(t_grid, dtype=float)
    if delay_bin_widths is not None:
        delay_bin_widths = np.asarray(delay_bin_widths, dtype=float)
        if delay_bin_widths.shape != t_grid.shape:
            raise ValueError("delay_bin_widths must have the same shape as t_grid")

    ok = np.isfinite(t_grid) & (t_grid > 0.0)
    if delay_bin_widths is not None:
        ok &= np.isfinite(delay_bin_widths) & (delay_bin_widths > 0.0)
        delay_bin_widths = delay_bin_widths[ok]
    t_grid = t_grid[ok]

    if len(t_grid) < 2:
        raise ValueError("Need at least 2 valid t-grid points.")

    if delay_bin_widths is not None:
        dt = delay_bin_widths
    else:
        # Backward-compatible approximation for older yield files that store
        # only bin centers. Newly generated files contain the exact widths.
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

    # print(f'tgrid: {t_grid}')

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

RUN_RES = (
    # Compact yield layout (current): logM_7.0_fEdd_0.1/
    re.compile(
        r"(?:^|/)M_(?P<M>[-+0-9.eE]+)_fEdd_"
        r"(?P<fedd>[-+0-9.eE]+)(?:/|$)"
    ),
    # Original simulation-output layout, retained for backwards compatibility.
    re.compile(
        r"(?:^|/)M_(?P<M>[-+0-9.eE]+)/fEdd_"
        r"(?P<fedd>[-+0-9.eE]+)(?:/|$)"
    ),
)


def parse_run_params_from_path(run_dir: Path):
    path = run_dir.as_posix()
    for pattern in RUN_RES:
        match = pattern.search(path)
        if match is not None:
            return float(match.group("M")), float(match.group("fedd"))
    raise ValueError(f"Could not parse M/fEdd from path: {run_dir}")


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
# ------------------ METALLICITY HELPERS ---------------------
# ============================================================

def mean_log_metallicity(zform: float, metallicity_model: str) -> float:
    """Return mu_Z = <log10(Z/Zsun)> at the formation redshift."""
    if metallicity_model not in VALID_METALLICITY_MODELS:
        raise ValueError(
            f"Unknown metallicity model {metallicity_model!r}; "
            f"choose from {sorted(VALID_METALLICITY_MODELS)}"
        )
    pars = METALLICITY_MODEL_PARAMETERS[metallicity_model]
    return float(np.log10(pars["a_Z"]) + zform * pars["b_Z"])


def metallicity_bin_weights(
    metallicities: np.ndarray,
    zform: float,
    metallicity_model: str,
    sigma_logz: float = SIGMA_LOGZ,
    z_sun: float = Z_SUN,
) -> tuple[np.ndarray, float]:
    """
    Integrate p(log10(Z/Zsun) | zform) over cells centred on the
    available fastcluster metallicity grid.

    The outer cells extend to +/- infinity, so the discrete weights sum
    to unity and the lowest/highest simulations absorb the distribution tails.
    """
    metallicities = np.asarray(metallicities, dtype=float)
    if metallicities.ndim != 1 or len(metallicities) == 0:
        raise ValueError("metallicities must be a non-empty 1D array")
    if np.any(~np.isfinite(metallicities)) or np.any(metallicities <= 0.0):
        raise ValueError("All metallicities must be finite and positive")
    if not np.isfinite(sigma_logz) or sigma_logz <= 0.0:
        raise ValueError("sigma_logz must be finite and positive")
    if not np.isfinite(z_sun) or z_sun <= 0.0:
        raise ValueError("z_sun must be finite and positive")

    x = np.log10(metallicities / z_sun)
    if len(np.unique(x)) != len(x):
        raise ValueError("Duplicate metallicities are not allowed")

    order = np.argsort(x)
    x_sorted = x[order]
    edges = np.empty(len(x_sorted) + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    if len(x_sorted) > 1:
        edges[1:-1] = 0.5 * (x_sorted[:-1] + x_sorted[1:])

    if metallicity_model == "single_Z":
        if len(metallicities) != 1:
            raise ValueError(
                "single_Z requires exactly one metallicity value"
            )
        return np.ones(1, dtype=float), float(x[0])

    mu_z = mean_log_metallicity(zform, metallicity_model)
    weights_sorted = np.diff(ndtr((edges - mu_z) / sigma_logz))
    weights_sorted = np.clip(weights_sorted, 0.0, 1.0)
    weights_sorted /= np.sum(weights_sorted)

    weights = np.empty_like(weights_sorted)
    weights[order] = weights_sorted
    return weights, mu_z


def discover_metallicity_run_bases(
    yields_base,
    metallicity_values=None,
) -> list[tuple[float, Path]]:
    """
    Resolve yield roots from a template containing ``{Z}``, e.g.
    ``/path/outputs/yields/Z{Z}``.

    If metallicity_values are omitted, existing paths matching the template
    are discovered automatically and the value replacing ``{Z}`` is parsed.
    """
    template = str(yields_base)

    if "{Z}" not in template:
        match = re.search(r"(?:yields|/Z)([-+0-9.eE]+)(?:/|$)", template)
        if match is None:
            raise ValueError(
                "--yields-base must contain a {Z} placeholder, e.g. "
                "'/path/outputs/yields/Z{Z}'"
            )
        z_value = float(match.group(1))
        if metallicity_values is not None:
            requested = [float(v) for v in metallicity_values]
            if len(requested) != 1 or not np.isclose(requested[0], z_value):
                raise ValueError(
                    "A concrete yield path can describe only one metallicity"
                )
        return [(z_value, Path(template))]

    resolved = []
    if metallicity_values:
        for z_label in metallicity_values:
            path = Path(template.format(Z=str(z_label)))
            if not path.exists():
                raise FileNotFoundError(f"Metallicity run root not found: {path}")
            resolved.append((float(z_label), path))
    else:
        path_pattern = template.replace("{Z}", "*")
        capture_re = re.compile(
            "^" + re.escape(template).replace(r"\{Z\}", r"(?P<Z>[-+0-9.eE]+)") + "$"
        )
        for match_path in sorted(glob.glob(path_pattern)):
            match = capture_re.match(match_path)
            if match is None:
                continue
            resolved.append((float(match.group("Z")), Path(match_path)))

    if not resolved:
        raise FileNotFoundError(
            f"No metallicity run roots found from template: {template}"
        )

    values = np.asarray([item[0] for item in resolved], dtype=float)
    if np.any(values <= 0.0) or len(np.unique(values)) != len(values):
        raise ValueError("Discovered metallicities must be positive and unique")
    return sorted(resolved, key=lambda item: item[0])


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
    metallicity_weight: float,
    z_bins: np.ndarray,
    mapping: dict,
):
    """
    Map a single run yield histogram into a contribution to K(z | z_form),
    using a precomputed common t -> z_merg mapping for this z_form.
    """
    df = read_table(yield_file)

    time_column = "t_delay_Gyr"
    yield_columns = (
        "Y_dt_delay_Gyr_given_M_fedd",  # new name
        "Y_tdelay_given_M_fedd",        # legacy name
    )

    if time_column not in df.columns:
        raise KeyError(f"{yield_file} must contain column '{time_column}'")

    yield_column = next((column for column in yield_columns if column in df.columns), None,)

    if yield_column is None:
        raise KeyError(
            f"{yield_file} must contain one of the yield columns: "
            + ", ".join(f"'{column}'" for column in yield_columns)        )

    t_all = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
    Y_all = pd.to_numeric( df[yield_column], errors="coerce").to_numpy(dtype=float)
    
    ref_t = mapping["reference_t_grid"]
    t_ok = np.isfinite(t_all) & (t_all > 0.0)
    t_clean = t_all[t_ok]

    # print(f't_clean: {t_clean}')
    # print(f'ref_t: {ref_t}')

    print(np.allclose(t_clean, ref_t, rtol=0, atol=1e-12))

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


    # Equation (7) of the accompanying paper
    # Time conversion: Y, t and dt are in Gyr, Gyr^-1 / Gyr * Gyr = Gyr^-1
    effective_rate_per_yr = (Y / t) * dt / GYR_TO_YR  # [per year]
    weight = (
        metallicity_weight
        * N_BH
        * pM
        * dlogM
        * pfedd
        * dfedd
        * effective_rate_per_yr
    )

    finite = np.isfinite(z_merg) & np.isfinite(weight) & (weight >= 0.0)
    z_merg = z_merg[finite]
    weight = weight[finite]

    hist, _ = np.histogram(z_merg, bins=z_bins, weights=weight)
    dz = np.diff(z_bins)
    K_bin = hist / dz

    return K_bin, {
        "int_Y_over_t_dt_per_yr": float(np.sum((Y / t) * dt) / GYR_TO_YR),
        "valid_bins": int(len(z_merg)),
    }



# ============================================================
# ------------------------- CORE RUN -------------------------
# ============================================================

def run_proto_mrd(
    yields_base,
    alpha,
    label,
    pm_file,
    pfedd_file,
    redshift_model,
    metallicity_model,
    zform,
    zmax=10.5,
    nz=200,
    SNR_lim=30,
    outdir="../protoMRD",
    yield_labels=None,
    cosmo_cache=None,
    metallicity_values=None,
):

    metallicity_roots = discover_metallicity_run_bases(
        yields_base=yields_base,
        metallicity_values=metallicity_values,
    )
    metallicities = np.asarray([item[0] for item in metallicity_roots], dtype=float)
    if metallicity_model == "single_Z" and len(metallicities) != 1:
        raise ValueError(
            "single_Z requires exactly one value via --metallicity-value"
        )
    pz_weights, mu_z = metallicity_bin_weights(
        metallicities=metallicities,
        zform=zform,
        metallicity_model=metallicity_model,
    )

    outdir = (
        Path(outdir)
        / f'SNRs_{SNR_lim}'
        / f"SG_alpha_{alpha}_{label}"
        / metallicity_model
        / redshift_model
        / f"{zform}"
    )
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

    summary_records = []
    for (metallicity, root), pz_weight in zip(metallicity_roots, pz_weights):
        alpha_root = root / f"SG_alpha_{alpha}"
        SNR_root = alpha_root / f'SNRs_30'
        model_root = SNR_root / label
        run_dirs = list(model_root.glob("M_*_fEdd_*"))
        if len(run_dirs) == 0:
            raise RuntimeError(
                f"No yield directories found under {model_root}; expected "
                f"M_<value>_fEdd_<value>/"
            )
        
        for run_dir in run_dirs:
            try:
                summary_file = find_summary_file(run_dir)
                summary_records.append({
                    "summary_file": summary_file,
                    "metallicity": metallicity,
                    "metallicity_weight": float(pz_weight),
                    "runs_root": str(root),
                })
            except FileNotFoundError:
                print(f"[warning] skipping run with no summary file: {run_dir}")
                continue

    if len(summary_records) == 0:
        raise FileNotFoundError(
            f"No usable summary files found for label={label}"
        )

    for record in summary_records:
        if yield_labels is not None:
            record["yield_labels"] = tuple(str(value) for value in yield_labels)
            continue

        summary_file = record["summary_file"]
        df_summary = read_table(summary_file)
        if "label" not in df_summary.columns:
            raise KeyError(f"No label column in {summary_file}")
        record["yield_labels"] = tuple(
            df_summary["label"].dropna().astype(str).unique()
        )

    common_mapping = None
    for record in summary_records:
        print(f'record: {record}')
        summary_file = record["summary_file"]
        run_dir = summary_file.parent
        for yl in ['1g']:
            yfile = run_dir / f"z_{zform}" / f"yield_{yl}.txt"
            if yfile.exists():
                print('common mapping generation...')
                try:
                    df_tmp = read_table(yfile)
                    if "t_delay_Gyr" not in df_tmp.columns:
                        continue
                    t_grid = pd.to_numeric(
                        df_tmp["t_delay_Gyr"], errors="coerce"
                    ).to_numpy(dtype=float)
                    delay_bin_widths = None
                    if "t_delay_bin_width_Gyr" in df_tmp.columns:
                        delay_bin_widths = pd.to_numeric(
                            df_tmp["t_delay_bin_width_Gyr"], errors="coerce"
                        ).to_numpy(dtype=float)

                    common_mapping = precompute_time_to_redshift_mapping(
                        t_grid=t_grid,
                        age_form=age_form,
                        age_today=age_today,
                        age_to_z=age_to_z,
                        delay_bin_widths=delay_bin_widths,
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

    for record in summary_records:
        summary_file = record["summary_file"]
        metallicity = record["metallicity"]
        metallicity_weight = record["metallicity_weight"]
        run_dir = summary_file.parent

        try:
            M, fedd = parse_run_params_from_path(run_dir)
            logM=np.log10(M)
        except Exception as e:
            print(f"[warning] skipping {run_dir}: could not parse M/fEdd ({e})")
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

        # print(f'pM_val:{pM_val}, pfedd_val: {pfedd_val}, dlogM: {dlogM}')

        for yl in ['1g']:
            this = df_summary

            if len(this) == 0:
                continue

            # print(f'yl: {yl}, this: {this}')

            yield_file = run_dir / f"z_{zform}" / f"yield_{yl}.txt"
            if not yield_file.exists():
                print(f"[warning] missing {yield_file}, skipping")
                continue

            if "N_BH" not in this.columns:
                print(f"[warning] N_BH column missing in {summary_file}, skipping {yl}")
                continue

            N_BH = float(pd.to_numeric(this["N_BH"], errors="coerce").mean())
            if not np.isfinite(N_BH) or N_BH <= 0.0:
                continue

            # print(f'Nbh: {N_BH}')
            # print(f'common mapping: {common_mapping}')


            try:
                K_bin, diag = build_run_contribution(
                    yield_file=yield_file,
                    N_BH=N_BH,
                    pM=pM_val,
                    pfedd=pfedd_val,
                    dlogM=dlogM,
                    dfedd=dfedd,
                    metallicity_weight=metallicity_weight,
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
                "metallicity": metallicity,
                "log10_Z_over_Zsun": np.log10(metallicity / Z_SUN),
                "metallicity_weight": metallicity_weight,
                "pM_given_zform": pM_val,
                "pfedd_given_M_zform": pfedd_val,
                "dlogM": dlogM,
                "dfEdd": dfedd,
                "pM_times_dlogM": pM_val * dlogM,
                "pfedd_times_dfEdd": pfedd_val * dfedd,
                "N_BH": N_BH,
                "int_Y_over_t_dt_per_yr": diag["int_Y_over_t_dt_per_yr"],
                "valid_delay_bins": diag["valid_bins"],
                "integrated_contribution": float(np.sum(K_bin * np.diff(z_bins))),
            })

            
    df_kernel = pd.DataFrame({
        "z": np.round(z_centers, 3),
        "K_proto": K_total,
    })
    df_kernel.to_csv(outdir / "kernel_vs_z.txt", sep="\t", index=False)

    df_contrib = pd.DataFrame(contribution_rows)
    df_contrib.to_csv(outdir / "contributions_by_run.txt", sep="\t", index=False)

    pd.DataFrame({
        "metallicity": metallicities,
        "log10_Z_over_Zsun": np.log10(metallicities / Z_SUN),
        "metallicity_weight": pz_weights,
    }).to_csv(outdir / "metallicity_weights.txt", sep="\t", index=False)

    meta = pd.DataFrame([{
        "redshift_model": redshift_model,
        "metallicity_model": metallicity_model,
        "a_Z": METALLICITY_MODEL_PARAMETERS.get(metallicity_model, {}).get("a_Z", np.nan),
        "b_Z": METALLICITY_MODEL_PARAMETERS.get(metallicity_model, {}).get("b_Z", np.nan),
        "mu_log10_Z_over_Zsun": mu_z,
        "sigma_log10_Z_over_Zsun": SIGMA_LOGZ,
        "Z_sun": Z_SUN,
        "metallicity_grid": ",".join(map(str, metallicities)),
        "metallicity_weights": ",".join(map(str, pz_weights)),
        "label": label,
        "z_form": zform,
        "zmin": z_bins[0],
        "zmax": z_bins[-1],
        "nz": nz,
        "nruns_used": len(df_contrib),
    }])
    meta.to_csv(outdir / "meta.txt", sep="\t", index=False)

    print(f"Done. Wrote proto-MRD kernel to {outdir}")
    

# ============================================================
# -------------------------- CLI -----------------------------
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--yields-base",
        dest="yields_base",
        required=True,
        help=(
            "Metallicity-dependent yield-root template, e.g. "
            "'/path/outputs/yields/Z{Z}'"
        ),
    )
    p.add_argument("--alpha", required=True, help="Alpha value used to select SG_alpha_<alpha>/")
    p.add_argument("--label", required=True,
                   help="Physical-model directory below SG_alpha_<alpha>")
    p.add_argument("--pm-file", required=True, help="Path to pM_given_z npz")
    p.add_argument("--pfedd-file", required=True, help="Path to pfedd_given_Mz npz")
    p.add_argument("--redshift-model", required=True, help="Label, e.g. SE or EL")
    p.add_argument(
        "--metallicity-model",
        required=True,
        choices=sorted(VALID_METALLICITY_MODELS),
        help="Nuclear metallicity evolution model",
    )
    p.add_argument(
        "--metallicity-values",
        nargs="+",
        default=None,
        help="Optional explicit Z grid; values must match the {Z} directory labels",
    )
    p.add_argument(
        "--metallicity-value",
        type=float,
        default=None,
        help="Single explicit Z value required when --metallicity-model single_Z",
    )
    p.add_argument("--zform", required=True, type=float, help="Formation redshift bin center")
    p.add_argument("--zmax", type=float, default=10.5)
    p.add_argument("--nz", type=int, default=200, help="Number of merger-z bins")
    p.add_argument("--SNR-lim", type=int, default=30)
    p.add_argument("--outdir", default="../outputs/protoMRD")
    p.add_argument(
        "--yield-labels",
        nargs="+",
        default=["1g", "ng"],
        help="Which yield files to include",
    )

    return p.parse_args()


def main():
    args = parse_args()
    metallicity_values = args.metallicity_values
    if args.metallicity_model == "single_Z":
        if args.metallicity_value is not None and metallicity_values is not None:
            raise ValueError("Use either --metallicity-value or --metallicity-values, not both")
        metallicity_values = (
            [args.metallicity_value]
            if args.metallicity_value is not None
            else metallicity_values
        )
        if metallicity_values is None or len(metallicity_values) != 1:
            raise ValueError(
                "single_Z requires exactly one value via --metallicity-value"
            )
    elif args.metallicity_value is not None:
        raise ValueError("--metallicity-value is only valid with --metallicity-model single_Z")

    run_proto_mrd(
        yields_base=args.yields_base,
        alpha=args.alpha,
        label=args.label,
        pm_file=args.pm_file,
        pfedd_file=args.pfedd_file,
        redshift_model=args.redshift_model,
        metallicity_model=args.metallicity_model,
        zform=args.zform,
        zmax=args.zmax,
        nz=args.nz,
        SNR_lim=args.SNR_lim,
        outdir=args.outdir,
        yield_labels=tuple(args.yield_labels),
        cosmo_cache=None,
        metallicity_values=metallicity_values,
    )


if __name__ == "__main__":
    main()
