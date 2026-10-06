#!/usr/bin/env python3
"""
Build p(M|z) on a regular (z, log10 M) grid and save it as .npz and/or .txt.

This follows the same underlying SMBH-mass model as the user's extraction code:
- for redshift == "zero": use the local empirical SMBH mass list
- otherwise: use the z-dependent broken power-law parameters stored in
  input/nAGN_models/M_SMBH_z_<redshift_key>.npz

Main output format (recommended):
    .npz with keys:
        z_grid       shape (Nz,)
        M_grid       shape (NM,)      # log10(M / Msun)
        pM_given_z   shape (Nz, NM)   # conditional PDF over log10 M

Optional text output:
    columns = z, log10M, pM_given_z

Note on convention:
This script returns p(log10 M | z), i.e. a density with respect to d log10 M.
That is usually the safest choice when your simulation grid is uniform in log-mass.

Example:
    python create_p_M.py \
        --redshift-key EL \
        --params-file input/nAGN_models/M_SMBH_z_EL.npz \
        --out-npz input/population_models/pM_given_z.npz \
        --out-txt input/population_models/pM_given_z.txt \
        --zmin 0 --zmax 10.5 --nz 150 \
        --Mmin 5.0 --Mmax 10.0 --dlogM 0.1

For the redshift="zero" case:
    python create_p_M.py \
        --redshift-key zero \
        --local-file input/SMBHmass_local_AGNlifetime_pairs.txt \
        --out-npz input/population_models/pM_given_z.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


LN10 = np.log(10.0)


def _safe_power_integral(x0, x1, slope):
    """
    Integral of x^(slope-1) dx from x0 to x1.
    Handles slope ~ 0 as the log-uniform limit.
    """
    x0, x1, slope = np.broadcast_arrays(
        np.asarray(x0, dtype=float),
        np.asarray(x1, dtype=float),
        np.asarray(slope, dtype=float),
    )

    out = np.empty_like(slope, dtype=float)
    small = np.isclose(slope, 0.0, atol=1e-12, rtol=0.0)

    out[small] = np.log(x1[small] / x0[small])
    out[~small] = (x1[~small] ** slope[~small] - x0[~small] ** slope[~small]) / slope[~small]
    return out


def _branch_pdf_linearM(M, xlo, xhi, slope):
    """
    Normalized truncated power-law branch on linear mass M:
        f(M) ∝ M^(slope - 1),  xlo <= M <= xhi

    Returns density with respect to dM.
    """
    M, xlo, xhi, slope = np.broadcast_arrays(
        np.asarray(M, dtype=float),
        np.asarray(xlo, dtype=float),
        np.asarray(xhi, dtype=float),
        np.asarray(slope, dtype=float),
    )

    norm = _safe_power_integral(xlo, xhi, slope)
    out = np.zeros_like(M, dtype=float)

    valid = (M >= xlo) & (M <= xhi) & np.isfinite(norm) & (norm > 0.0)
    small = np.isclose(slope, 0.0, atol=1e-12, rtol=0.0)

    mask1 = valid & small
    mask2 = valid & ~small

    out[mask1] = 1.0 / (M[mask1] * norm[mask1])
    out[mask2] = (M[mask2] ** (slope[mask2] - 1.0)) / norm[mask2]

    return out


def broken_powerlaw_pdf_logM(
    logM_grid: np.ndarray,
    logMmin: float,
    logMbrk: float,
    logMmax: np.ndarray,
    slopelo: np.ndarray,
    slopehi: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-broken power law matching the sampling logic in sample_broken_powerlaw().

    Output is p(log10 M), i.e. density with respect to d log10 M.
    For a fixed z:
        p(log10 M | z) = M ln(10) * p(M | z)

    We choose mixture weights so the PDF is continuous at the break.
    """
    logM_grid = np.asarray(logM_grid, dtype=float)
    M = 10.0 ** logM_grid

    Mmin = 10.0 ** float(logMmin)
    Mbrk = 10.0 ** float(logMbrk)
    Mmax = 10.0 ** np.asarray(logMmax, dtype=float)
    slopelo = np.asarray(slopelo, dtype=float)
    slopehi = np.asarray(slopehi, dtype=float)

    if np.any(Mmax <= Mbrk) or np.any(~np.isfinite(Mmax)):
        raise ValueError("All Mmax values must be finite and strictly larger than the break mass.")

    # Individually normalized branch PDFs in linear mass
    f1 = _branch_pdf_linearM(M[None, :], Mmin, Mbrk, slopelo[:, None])
    f2 = _branch_pdf_linearM(M[None, :], Mbrk, Mmax[:, None], slopehi[:, None])

    # Values at the break used to impose continuity in the mixture
    I1 = _safe_power_integral(np.full_like(slopelo, Mmin), np.full_like(slopelo, Mbrk), slopelo)
    I2 = _safe_power_integral(np.full_like(slopehi, Mbrk), Mmax, slopehi)

    p1 = np.empty_like(slopelo, dtype=float)
    p2 = np.empty_like(slopehi, dtype=float)

    small1 = np.isclose(slopelo, 0.0, atol=1e-12, rtol=0.0)
    small2 = np.isclose(slopehi, 0.0, atol=1e-12, rtol=0.0)

    p1[small1] = 1.0 / (Mbrk * I1[small1])
    p1[~small1] = (Mbrk ** (slopelo[~small1] - 1.0)) / I1[~small1]

    p2[small2] = 1.0 / (Mbrk * I2[small2])
    p2[~small2] = (Mbrk ** (slopehi[~small2] - 1.0)) / I2[~small2]

    # Continuity condition:
    #   w_low * f1(Mbrk) = w_high * f2(Mbrk)
    # with w_low + w_high = 1
    w_low = p2 / (p1 + p2)
    w_high = p1 / (p1 + p2)

    pdf_M = w_low[:, None] * f1 + w_high[:, None] * f2
    pdf_logM = pdf_M * M[None, :] * LN10

    # Final numerical normalization over dlogM
    norm = np.trapz(pdf_logM, x=logM_grid, axis=1)
    bad = (~np.isfinite(norm)) | (norm <= 0.0)
    if np.any(bad):
        raise ValueError("Encountered invalid normalization while building p(logM|z).")
    pdf_logM /= norm[:, None]

    return pdf_logM


def empirical_pdf_logM_from_samples(
    logM_grid: np.ndarray,
    samples_linearM: np.ndarray,
) -> np.ndarray:
    """
    Build p(log10 M) from empirical samples using a histogram on a regular logM grid.
    """
    samples_linearM = np.asarray(samples_linearM, dtype=float)
    samples_linearM = samples_linearM[np.isfinite(samples_linearM) & (samples_linearM > 0.0)]
    if samples_linearM.size == 0:
        raise ValueError("No valid positive SMBH masses found in local empirical sample.")

    log_samples = np.log10(samples_linearM)

    # Build edges from centers
    dlogM = np.diff(logM_grid)
    if not np.allclose(dlogM, dlogM[0], rtol=1e-8, atol=1e-12):
        raise ValueError("For the empirical histogram mode, logM_grid must be uniformly spaced.")
    step = dlogM[0]
    edges = np.concatenate(([logM_grid[0] - 0.5 * step], logM_grid + 0.5 * step))

    hist, _ = np.histogram(log_samples, bins=edges, density=True)
    hist = hist.astype(float)

    norm = np.trapz(hist, x=logM_grid)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Could not normalize empirical p(logM|z).")
    hist /= norm
    return hist[None, :]


def load_redshift_dependent_params(params_file: str | Path, redshift_key: str):
    params = np.load(params_file)
    z_grid = np.asarray(params["z_grid"], dtype=float)

    if "EL" in redshift_key:
        A1_grid = np.asarray(params["A1_Edd"], dtype=float)
        A2_grid = np.asarray(params["A2_Edd"], dtype=float)
        Mmax_grid = np.asarray(params["Mmax_Edd"], dtype=float)
    elif "SE" in redshift_key:
        A1_grid = np.asarray(params["A1_merger"], dtype=float)
        A2_grid = np.asarray(params["A2_merger"], dtype=float)
        Mmax_grid = np.asarray(params["Mmax_merger"], dtype=float)
    else:
        raise ValueError(
            f"redshift_key={redshift_key!r} is not understood. "
            "Expected 'zero' or a key containing 'EL' or 'SE'."
        )

    return z_grid, A1_grid, A2_grid, Mmax_grid


def build_pM_given_z(
    redshift_key: str,
    logM_grid: np.ndarray,
    z_out: np.ndarray,
    logMmin: float,
    logMbrk: float,
    params_file: str | Path | None = None,
    local_file: str | Path | None = None,
    local_file_skip_header: int = 3,
    local_file_mass_col: int = 0,
    local_mass_max: float | None = None,
):
    """
    Return:
        z_grid_out   shape (Nz,)
        M_grid_out   shape (NM,)      [log10(M/Msun)]
        pM_given_z   shape (Nz, NM)   [density wrt dlog10M]
    """
    redshift_key = str(redshift_key)

    if redshift_key == "zero":
        if local_file is None:
            raise ValueError("For redshift_key='zero', you must provide --local-file.")

        masses = np.genfromtxt(
            local_file,
            usecols=(local_file_mass_col,),
            skip_header=local_file_skip_header,
            unpack=True,
        )
        masses = np.asarray(masses, dtype=float)

        if local_mass_max is not None:
            masses = masses[masses <= local_mass_max]

        pM = empirical_pdf_logM_from_samples(logM_grid, masses)
        z_grid_out = np.array([0.0], dtype=float)
        return z_grid_out, logM_grid, pM

    if params_file is None:
        raise ValueError("For redshift-dependent cases, you must provide --params-file.")

    z_model, A1_grid, A2_grid, Mmax_grid = load_redshift_dependent_params(params_file, redshift_key)

    A1_interp = np.interp(z_out, z_model, A1_grid)
    A2_interp = np.interp(z_out, z_model, A2_grid)
    Mmax_interp = np.interp(z_out, z_model, Mmax_grid)

    pM = broken_powerlaw_pdf_logM(
        logM_grid=logM_grid,
        logMmin=logMmin,
        logMbrk=logMbrk,
        logMmax=Mmax_interp,
        slopelo=A1_interp,
        slopehi=A2_interp,
    )

    return np.asarray(z_out, dtype=float), logM_grid, pM


def save_txt_triplets(outfile: str | Path, z_grid: np.ndarray, logM_grid: np.ndarray, pM_given_z: np.ndarray):
    rows = []
    for i, z in enumerate(z_grid):
        zz = np.full_like(logM_grid, z, dtype=float)
        rows.append(np.column_stack([zz, logM_grid, pM_given_z[i]]))
    arr = np.vstack(rows)
    header = (
        "z log10M pM_given_z\n"
        "Columns:\n"
        "  z          redshift\n"
        "  log10M     log10(M/Msun)\n"
        "  pM_given_z conditional PDF density wrt dlog10M"
    )
    np.savetxt(outfile, arr, header=header)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--redshift-key", required=True,
                   help="Either 'zero' or a key like EL / SE.")

    p.add_argument("--params-file", default=None,
                   help="Path to M_SMBH_z_<redshift_key>.npz for non-zero redshift cases.")
    p.add_argument("--local-file", default=None,
                   help="Path to SMBHmass_local_AGNlifetime_pairs.txt for the zero-redshift case.")

    p.add_argument("--out-npz", required=True,
                   help="Output .npz filename.")
    p.add_argument("--out-txt", default=None,
                   help="Optional output .txt filename.")

    p.add_argument("--zmin", type=float, default=0.0)
    p.add_argument("--zmax", type=float, default=10.5)
    p.add_argument("--nz", type=int, default=150)

    p.add_argument("--Mmin", type=float, default=5.0,
                   help="Minimum log10(M/Msun) for the output grid and broken-power-law lower cutoff.")
    p.add_argument("--Mmax", type=float, default=10.0,
                   help="Maximum log10(M/Msun) for the output grid.")
    p.add_argument("--dlogM", type=float, default=0.1,
                   help="Spacing of the output log10(M/Msun) grid.")
    p.add_argument("--Mbreak", type=float, default=7.4,
                   help="Break log10(M/Msun).")

    p.add_argument("--local-file-skip-header", type=int, default=3)
    p.add_argument("--local-file-mass-col", type=int, default=0)
    p.add_argument("--local-mass-max", type=float, default=None,
                   help="Optional max linear mass cutoff for the zero-redshift empirical sample.")

    return p.parse_args()


def main():
    args = parse_args()

    logM_grid = np.arange(args.Mmin, args.Mmax + 0.5 * args.dlogM, args.dlogM, dtype=float)
    z_out = np.linspace(args.zmin, args.zmax, args.nz, dtype=float)

    z_grid, M_grid, pM_given_z = build_pM_given_z(
        redshift_key=args.redshift_key,
        logM_grid=logM_grid,
        z_out=z_out,
        logMmin=args.Mmin,
        logMbrk=args.Mbreak,
        params_file=args.params_file,
        local_file=args.local_file,
        local_file_skip_header=args.local_file_skip_header,
        local_file_mass_col=args.local_file_mass_col,
        local_mass_max=args.local_mass_max,
    )

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, z_grid=z_grid, M_grid=M_grid, pM_given_z=pM_given_z)

    if args.out_txt is not None:
        out_txt = Path(args.out_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        save_txt_triplets(out_txt, z_grid, M_grid, pM_given_z)

    # Lightweight sanity report
    norms = np.trapz(pM_given_z, x=M_grid, axis=1)
    print(f"Saved NPZ to: {out_npz}")
    if args.out_txt is not None:
        print(f"Saved TXT to: {args.out_txt}")
    print(f"z_grid shape     = {z_grid.shape}")
    print(f"M_grid shape     = {M_grid.shape}")
    print(f"pM_given_z shape = {pM_given_z.shape}")
    print(f"Normalization range over dlog10M: [{norms.min():.6f}, {norms.max():.6f}]")


if __name__ == "__main__":
    main()
