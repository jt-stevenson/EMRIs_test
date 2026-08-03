#!/usr/bin/env python3
"""
Integrate proto-MRD kernels over AGN abundance:
    R(z) = ∫ dz_form K(z | z_form) * nAGN(z_form)

Also integrates the precomputed proto-level Mchirp distributions over z_form,
then saves final Mchirp percentiles as a function of merger redshift.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re

import astropy.units as u
from astropy.cosmology import Planck18

VALID_REDSHIFT_MODELS = {"SE", "EL"}
VALID_ABUNDANCE_MODELS = {"LAM", "HAM"}
DEFAULT_MCHIRP_PERCENTILES = (5.0, 16.0, 50.0, 84.0, 95.0)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--proto-dir",
        required=True,
        help="Base directory containing protoMRD kernels",
    )
    p.add_argument(
        "--redshift-model",
        required=True,
        choices=sorted(VALID_REDSHIFT_MODELS),
        help="Kernel family to use: SE or EL",
    )
    p.add_argument(
        "--AGN-abundance-model",
        required=True,
        choices=sorted(VALID_ABUNDANCE_MODELS),
        help="AGN abundance model: LAM or HAM",
    )

    p.add_argument(
        "--alpha",
        default="0.01",
        help="Viscosity parameter",
    )

    p.add_argument(
        "--label",
        default="G24_K18-3bb_0.0-IG25-agnostic-tau_x_1.",
        help="Physical model used in the corresponding fastcluster simulation",
    )

    p.add_argument(
        "--nagn-dir",
        default="../input/nAGN_models",
        help="Directory containing nAGN files",
    )
    p.add_argument(
        "--nagn-file",
        default=None,
        help=(
            "Optional explicit path to nAGN file. "
            "If omitted, uses <nagn-dir>/nAGN_<redshift-model>_<AGN-abundance-model>.txt"
        ),
    )

    p.add_argument(
        "--kernel-filename",
        default="kernel_vs_z.txt",
        help="Name of the kernel file inside each zform directory",
    )
    p.add_argument(
        "--proto-mchirp-filename",
        default="mchirp_distribution_vs_z.npz",
        help="Name of the proto-level Mchirp distribution file inside each zform directory",
    )
    p.add_argument(
        "--outdir",
        default="../outputs/MRD_results",
        help="Output directory",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip malformed/missing kernel files instead of failing",
    )
    p.add_argument(
        "--mchirp-percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_MCHIRP_PERCENTILES),
        help="Percentiles to save for Mchirp(z)",
    )

    return p.parse_args()


def read_two_column_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", comment="#")
    except Exception as e:
        raise RuntimeError(f"Could not read file {path}: {e}")

    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")

    x = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) == 0:
        raise ValueError(f"{path} contains no valid numeric rows")

    order = np.argsort(x)
    return x[order], y[order]


def read_kernel_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(path, sep=r"\s+|\t+", engine="python", comment="#")
    except Exception as e:
        raise RuntimeError(f"Could not read kernel file {path}: {e}")

    cols = list(df.columns)

    if "z" in cols and "K_proto" in cols:
        z = pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float)
        k = pd.to_numeric(df["K_proto"], errors="coerce").to_numpy(dtype=float)
    elif df.shape[1] >= 2:
        z = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        k = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError(
            f"{path} must contain columns 'z' and 'K_proto', or at least two columns"
        )

    ok = np.isfinite(z) & np.isfinite(k)
    z = z[ok]
    k = k[ok]

    if len(z) == 0:
        raise ValueError(f"{path} contains no valid kernel rows")

    order = np.argsort(z)
    return z[order], k[order]


def load_proto_mchirp_distribution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    required = {"z_centers", "mchirp_edges_Msun", "mass_by_z"}
    missing = required.difference(d.files)
    if missing:
        raise KeyError(f"{path} missing arrays: {sorted(missing)}")

    z = np.asarray(d["z_centers"], dtype=float)
    m_edges = np.asarray(d["mchirp_edges_Msun"], dtype=float)
    mass_by_z = np.asarray(d["mass_by_z"], dtype=float)

    if mass_by_z.ndim != 2:
        raise ValueError(f"mass_by_z in {path} must be 2D")
    if mass_by_z.shape[0] != len(z):
        raise ValueError(f"mass_by_z first dimension inconsistent with z_centers in {path}")
    if mass_by_z.shape[1] != len(m_edges) - 1:
        raise ValueError(f"mass_by_z second dimension inconsistent with mchirp_edges in {path}")

    return z, m_edges, mass_by_z


def load_all_kernels(proto_dir: Path, redshift_model: str, kernel_filename: str, allow_missing: bool):
    root = proto_dir / redshift_model
    if not root.exists():
        raise FileNotFoundError(f"Kernel directory not found: {root}")

    zform_dirs = [d for d in root.iterdir() if d.is_dir()]
    if len(zform_dirs) == 0:
        raise FileNotFoundError(f"No zform subdirectories found in {root}")

    rows = []
    z_ref = None

    for d in sorted(zform_dirs, key=lambda x: float(x.name)):
        try:
            zform = float(d.name)
        except ValueError:
            if allow_missing:
                print(f"[warning] skipping non-numeric zform directory: {d}")
                continue
            raise ValueError(f"Non-numeric zform directory name: {d.name}")

        kfile = d / kernel_filename
        if not kfile.exists():
            msg = f"Missing kernel file: {kfile}"
            if allow_missing:
                print(f"[warning] {msg}")
                continue
            raise FileNotFoundError(msg)

        try:
            z, k = read_kernel_file(kfile)
        except Exception as e:
            if allow_missing:
                print(f"[warning] skipping {kfile}: {e}")
                continue
            raise

        if z_ref is None:
            z_ref = z
        else:
            if len(z) != len(z_ref) or not np.allclose(z, z_ref, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Inconsistent merger-z grid in {kfile}. "
                    f"All kernel_vs_z.txt files must share the same z grid."
                )

        rows.append((zform, k))

    if len(rows) == 0:
        raise RuntimeError(f"No usable kernel files found in {root}")

    rows.sort(key=lambda x: x[0])
    zform_grid = np.array([r[0] for r in rows], dtype=float)
    K_matrix = np.vstack([r[1] for r in rows])

    return z_ref, zform_grid, K_matrix


def load_all_proto_mchirp_distributions(
    proto_dir: Path,
    redshift_model: str,
    proto_mchirp_filename: str,
    allow_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = proto_dir / redshift_model
    if not root.exists():
        raise FileNotFoundError(f"Kernel directory not found: {root}")

    zform_dirs = [d for d in root.iterdir() if d.is_dir()]
    if len(zform_dirs) == 0:
        raise FileNotFoundError(f"No zform subdirectories found in {root}")

    rows = []
    z_ref = None
    m_edges_ref = None

    for d in sorted(zform_dirs, key=lambda x: float(x.name)):
        try:
            zform = float(d.name)
        except ValueError:
            if allow_missing:
                print(f"[warning] skipping non-numeric zform directory: {d}")
                continue
            raise ValueError(f"Non-numeric zform directory name: {d.name}")

        mfile = d / proto_mchirp_filename
        if not mfile.exists():
            msg = f"Missing proto Mchirp file: {mfile}"
            if allow_missing:
                print(f"[warning] {msg}")
                continue
            raise FileNotFoundError(msg)

        try:
            z, m_edges, mass_by_z = load_proto_mchirp_distribution(mfile)
        except Exception as e:
            if allow_missing:
                print(f"[warning] skipping {mfile}: {e}")
                continue
            raise

        if z_ref is None:
            z_ref = z
        else:
            if len(z) != len(z_ref) or not np.allclose(z, z_ref, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Inconsistent merger-z grid in {mfile}. "
                    f"All proto Mchirp files must share the same z grid."
                )

        if m_edges_ref is None:
            m_edges_ref = m_edges
        else:
            if len(m_edges) != len(m_edges_ref) or not np.allclose(m_edges, m_edges_ref, rtol=0, atol=1e-12):
                raise ValueError(
                    f"Inconsistent Mchirp bin edges in {mfile}. "
                    f"All proto Mchirp files must share the same mass grid."
                )

        rows.append((zform, mass_by_z))

    if len(rows) == 0:
        raise RuntimeError(f"No usable proto Mchirp files found in {root}")

    rows.sort(key=lambda x: x[0])
    zform_grid = np.array([r[0] for r in rows], dtype=float)
    mass_tensor = np.stack([r[1] for r in rows], axis=0)
    return z_ref, zform_grid, m_edges_ref, mass_tensor


def extract_tau_multiplier(label):
    match = re.search(r"tau_x_([0-9.]+)", label)
    if match is None:
        raise ValueError(f"Cannot find tau_x multiplier in label: {label}")
    return float(match.group(1))


def abs_dt_dz_yr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    Hz = Planck18.H(z).to_value(1 / u.yr)
    return 1.0 / ((1.0 + z) * Hz)


def abs_dz_dt_per_yr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    Hz = Planck18.H(z).to_value(1 / u.yr)
    return (1.0 + z) * Hz


def interpolate_nagn_to_zform(zform_grid: np.ndarray, z_nagn: np.ndarray, nagn: np.ndarray) -> np.ndarray:
    return np.interp(zform_grid, z_nagn, nagn, left=0.0, right=0.0)


def integrate_over_zform(
    zform_grid: np.ndarray,
    K_matrix: np.ndarray,
    nagn_on_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prefactor = abs_dt_dz_yr(zform_grid) * nagn_on_grid
    integrand = K_matrix * prefactor[:, None]
    Kz_total = np.trapz(integrand, x=zform_grid, axis=0)
    return Kz_total, prefactor


def integrate_mchirp_over_zform(
    zform_grid: np.ndarray,
    mass_tensor: np.ndarray,
    nagn_on_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prefactor = abs_dt_dz_yr(zform_grid) * nagn_on_grid
    integrand = mass_tensor * prefactor[:, None, None]
    mass_by_z_total = np.trapz(integrand, x=zform_grid, axis=0)
    return mass_by_z_total, prefactor


def weighted_percentiles_from_hist(m_edges: np.ndarray, mass_hist: np.ndarray, percentiles) -> np.ndarray:
    m_edges = np.asarray(m_edges, dtype=float)
    mass_hist = np.asarray(mass_hist, dtype=float)
    q = np.asarray(percentiles, dtype=float)

    if mass_hist.ndim != 1:
        raise ValueError("mass_hist must be 1D")
    if len(m_edges) != len(mass_hist) + 1:
        raise ValueError("m_edges must have length len(mass_hist)+1")

    widths = np.diff(m_edges)
    weights = mass_hist * widths

    good = np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return np.full(len(q), np.nan)

    weights = weights[good]
    left = m_edges[:-1][good]
    right = m_edges[1:][good]

    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(q), np.nan)

    cdf_hi = np.cumsum(weights) / total
    cdf_lo = np.concatenate(([0.0], cdf_hi[:-1]))

    out = np.full(len(q), np.nan)
    for i, qq in enumerate(q / 100.0):
        idx = np.searchsorted(cdf_hi, qq, side="left")
        idx = min(max(idx, 0), len(weights) - 1)

        if cdf_hi[idx] <= cdf_lo[idx] + 1e-15:
            out[i] = 0.5 * (left[idx] + right[idx])
        else:
            frac = (qq - cdf_lo[idx]) / (cdf_hi[idx] - cdf_lo[idx])
            frac = np.clip(frac, 0.0, 1.0)
            out[i] = left[idx] + frac * (right[idx] - left[idx])

    return out


def main():
    args = parse_args()

    combo_name = f"{args.redshift_model}_{args.AGN_abundance_model}"

    proto_dir = Path(args.proto_dir) / f"alpha_{args.alpha}" / args.label
    nagn_dir = Path(args.nagn_dir)
    outdir = Path(args.outdir) / f"alpha_{args.alpha}" / args.label / combo_name
    outdir.mkdir(parents=True, exist_ok=True)

    if args.nagn_file is not None:
        nagn_file = Path(args.nagn_file)
    else:
        nagn_file = nagn_dir / f"nAGN_{combo_name}.txt"

    if not nagn_file.exists():
        raise FileNotFoundError(
            f"nAGN file not found: {nagn_file}\n"
            f"Either provide --nagn-file explicitly or place the file there."
        )

    z_grid, zform_grid, K_matrix = load_all_kernels(
        proto_dir=proto_dir,
        redshift_model=args.redshift_model,
        kernel_filename=args.kernel_filename,
        allow_missing=args.allow_missing,
    )

    z_nagn, nagn = read_two_column_file(nagn_file)
    nagn_on_zform = interpolate_nagn_to_zform(zform_grid, z_nagn, nagn)

    Kz_total, zform_prefactor = integrate_over_zform(
        zform_grid=zform_grid,
        K_matrix=K_matrix,
        nagn_on_grid=nagn_on_zform,
    )

    Rz = abs_dz_dt_per_yr(z_grid) * Kz_total

    df_out = pd.DataFrame({
        "z": z_grid,
        "R_of_z": Rz,
    })
    df_out.to_csv(outdir / "MRD_vs_z.txt", sep="\t", index=False)

    contrib = np.trapz(K_matrix, x=z_grid, axis=1) * zform_prefactor
    df_diag = pd.DataFrame({
        "z_form": zform_grid,
        "nAGN_interp": nagn_on_zform,
        "abs_dt_dz_form_yr": abs_dt_dz_yr(zform_grid),
        "zform_prefactor": zform_prefactor,
        "integrated_kernel_contribution": contrib,
    })
    df_diag.to_csv(outdir / "zform_contributions.txt", sep="\t", index=False)

    # Final Mchirp(z) distribution and percentiles
    try:
        z_mchirp, zform_mchirp, mchirp_edges, mass_tensor = load_all_proto_mchirp_distributions(
            proto_dir=proto_dir,
            redshift_model=args.redshift_model,
            proto_mchirp_filename=args.proto_mchirp_filename,
            allow_missing=args.allow_missing,
        )

        if len(z_mchirp) != len(z_grid) or not np.allclose(z_mchirp, z_grid, rtol=0, atol=1e-12):
            raise ValueError("Proto Mchirp z grid does not match kernel z grid")
        if len(zform_mchirp) != len(zform_grid) or not np.allclose(zform_mchirp, zform_grid, rtol=0, atol=1e-12):
            raise ValueError("Proto Mchirp z_form grid does not match kernel z_form grid")

        mass_by_z_total, _ = integrate_mchirp_over_zform(
            zform_grid=zform_grid,
            mass_tensor=mass_tensor,
            nagn_on_grid=nagn_on_zform,
        )

        q_list = np.asarray(args.mchirp_percentiles, dtype=float)
        rows = []
        for iz, zval in enumerate(z_grid):
            mass_hist = mass_by_z_total[iz]
            qvals = weighted_percentiles_from_hist(mchirp_edges, mass_hist, q_list)
            row = {
                "z": float(zval),
                "mass_weight_total": float(np.sum(mass_hist * np.diff(mchirp_edges))),
            }
            for q, qv in zip(q_list, qvals):
                row[f"mchirp_p{str(q).replace('.', 'p')}_Msun"] = float(qv) if np.isfinite(qv) else np.nan
            rows.append(row)

        pd.DataFrame(rows).to_csv(outdir / "mchirp_percentiles_vs_z.txt", sep="\t", index=False)
        np.savez_compressed(
            outdir / "mchirp_distribution_vs_z.npz",
            z=z_grid,
            mchirp_edges_Msun=mchirp_edges,
            mchirp_centers_Msun=0.5 * (mchirp_edges[:-1] + mchirp_edges[1:]),
            mass_by_z=mass_by_z_total,
        )

        mchirp_zform_budget = np.trapz(
            np.trapz(mass_tensor, x=z_grid, axis=1),
            x=mchirp_edges[:-1],
            axis=1,
        ) * zform_prefactor
        pd.DataFrame({
            "z_form": zform_grid,
            "nAGN_interp": nagn_on_zform,
            "abs_dt_dz_form_yr": abs_dt_dz_yr(zform_grid),
            "zform_prefactor": zform_prefactor,
            "integrated_mchirp_contribution": mchirp_zform_budget,
        }).to_csv(outdir / "zform_mchirp_contributions.txt", sep="\t", index=False)

        has_mchirp = True
    except Exception as e:
        if args.allow_missing:
            print(f"[warning] could not build final Mchirp(z) summary: {e}")
            has_mchirp = False
        else:
            raise

    meta = pd.DataFrame([{
        "redshift_model": args.redshift_model,
        "AGN_abundance_model": args.AGN_abundance_model,
        "combo_name": combo_name,
        "proto_dir": str(proto_dir),
        "nagn_file": str(nagn_file),
        "n_z_bins": len(z_grid),
        "n_zform_bins": len(zform_grid),
        "z_min": float(np.min(z_grid)),
        "z_max": float(np.max(z_grid)),
        "zform_min": float(np.min(zform_grid)),
        "zform_max": float(np.max(zform_grid)),
        "has_mchirp_summary": bool(has_mchirp),
        "mchirp_percentiles": ",".join(map(str, args.mchirp_percentiles)),
    }])
    meta.to_csv(outdir / "meta.txt", sep="\t", index=False)

    Hz_z = Planck18.H(z_grid).to_value(1 / u.yr)
    abs_dt_dz_z = 1.0 / ((1.0 + z_grid) * Hz_z)

    Ntot_from_R = np.trapz(Rz * abs_dt_dz_z, x=z_grid)
    Ntot_from_K = np.trapz(Kz_total, x=z_grid)
    kernel_integral_per_zform = np.trapz(K_matrix, x=z_grid, axis=1)
    Ntot_from_zform_budget = np.trapz(
        zform_prefactor * kernel_integral_per_zform,
        x=zform_grid
    )

    print("\nConsistency checks:")
    print(f"  ∫ R_source(z) |dt/dz| dz   = {Ntot_from_R:.6e}")
    print(f"  ∫ K_total(z) dz            = {Ntot_from_K:.6e}")
    print(f"  ∫ dz_form pref * ∫K dz     = {Ntot_from_zform_budget:.6e}")

    if Ntot_from_R > 0:
        rel1 = abs(Ntot_from_R - Ntot_from_K) / Ntot_from_R
        rel2 = abs(Ntot_from_R - Ntot_from_zform_budget) / Ntot_from_R
        print(f"  relative mismatch (R vs K)       = {rel1:.3e}")
        print(f"  relative mismatch (R vs budget)  = {rel2:.3e}")

    print("Done.")
    print(f"  Model combination : {combo_name}")
    print(f"  Output written to : {outdir}")
    print(f"  Main file         : {outdir / 'MRD_vs_z.txt'}")
    if has_mchirp:
        print(f"  Mchirp file       : {outdir / 'mchirp_percentiles_vs_z.txt'}")


if __name__ == "__main__":
    main()
