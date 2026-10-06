#!/usr/bin/env python3
"""
Integrate proto-MRD kernels over AGN abundance:
    R(z) = ∫ dz_form K(z | z_form) * nAGN(z_form)

Each proto-MRD family has already been mixed over the fastcluster metallicity
grid according to flat_Z or evolving_Z, or uses one explicit metallicity for
the single_Z prescription.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.cosmology import Planck18

VALID_REDSHIFT_MODELS = {"SE", "EL"}
VALID_ABUNDANCE_MODELS = {"LAM", "HAM"}
VALID_METALLICITY_MODELS = {"flat_Z", "evolving_Z", "single_Z"}

GPC3_TO_MPC3 = 1.0e9


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
        "--metallicity-model",
        required=True,
        choices=sorted(VALID_METALLICITY_MODELS),
        help="Metallicity prescription used for the proto-kernel family",
    )
    p.add_argument(
        "--metallicity-value",
        type=float,
        default=None,
        help="Single Z value used for the single_Z output subfolder",
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
        "--outdir",
        default="../MRD_results",
        help="Output directory",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip malformed/missing kernel files instead of failing",
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


def main():
    args = parse_args()

    if args.metallicity_model == "single_Z":
        if args.metallicity_value is None:
            raise ValueError(
                "single_Z requires --metallicity-value"
            )
        metallicity_subdir = f"Z_{args.metallicity_value:g}"
    else:
        if args.metallicity_value is not None:
            raise ValueError(
                "--metallicity-value is only valid with --metallicity-model single_Z"
            )
        metallicity_subdir = None

    combo_name = (
        f"{args.redshift_model}_{args.AGN_abundance_model}_{args.metallicity_model}"
    )

    proto_dir = (
        Path(args.proto_dir)
        / f"SG_alpha_{args.alpha}_{args.label}"
        / args.metallicity_model
    )
    nagn_dir = Path(args.nagn_dir)
    outdir = (
        Path(args.outdir)
        / f"SG_alpha_{args.alpha}_{args.label}"
    )
    if metallicity_subdir is not None:
        outdir /= metallicity_subdir
    else:
        outdir /= args.metallicity_model
    outdir /= f"{args.redshift_model}_{args.AGN_abundance_model}"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.nagn_file is not None:
        nagn_file = Path(args.nagn_file)
    else:
        nagn_combo_name = f"{args.redshift_model}_{args.AGN_abundance_model}"
        nagn_file = nagn_dir / f"nAGN_{nagn_combo_name}.txt"

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
    nagn = nagn * GPC3_TO_MPC3  # convert from Mpc^-3 to Gpc^-3
    nagn_on_zform = interpolate_nagn_to_zform(zform_grid, z_nagn, nagn)

    Kz_total, zform_prefactor = integrate_over_zform(
        zform_grid=zform_grid,
        K_matrix=K_matrix,
        nagn_on_grid=nagn_on_zform,
    )

    Rz = abs_dz_dt_per_yr(z_grid) * Kz_total

    df_out = pd.DataFrame({
        "z": z_grid,
        "MRD_Gpc^-3_yr^-1": Rz,
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

    meta = pd.DataFrame([{
        "redshift_model": args.redshift_model,
        "AGN_abundance_model": args.AGN_abundance_model,
        "metallicity_model": args.metallicity_model,
        "combo_name": combo_name,
        "proto_dir": str(proto_dir),
        "nagn_file": str(nagn_file),
        "n_z_bins": len(z_grid),
        "n_zform_bins": len(zform_grid),
        "z_min": float(np.min(z_grid)),
        "z_max": float(np.max(z_grid)),
        "zform_min": float(np.min(zform_grid)),
        "zform_max": float(np.max(zform_grid)),
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


if __name__ == "__main__":
    main()
