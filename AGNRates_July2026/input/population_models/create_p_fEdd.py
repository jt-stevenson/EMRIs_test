#!/usr/bin/env python3
"""
Convert a deterministic fEdd(M,z) field into a discrete p(fEdd | M, z) file
compatible with assign_redshift_to_merger_events.py.

Input NPZ format expected:
    scenario   : optional string
    z_grid     : shape (Nz,)
    M_grid     : shape (NM,)          # log10(M/Msun)
    fEdd_grid  : shape (Nz, NM)       # deterministic fEdd value at each (z, M)

Output NPZ format written:
    z_grid           : shape (Nz,)
    M_grid           : shape (NM,)
    fedd_grid        : shape (Nf,)
    pfedd_given_Mz   : shape (Nz, NM, Nf)

Interpretation:
    pfedd_given_Mz[i, j, k] = p(fedd_grid[k] | M_grid[j], z_grid[i])

For each (z_i, M_j), all probability is placed in the closest matching
fEdd bin in fEdd_grid.

This is exactly the format read by load_pfEdd_given_Mz() in
assign_redshift_to_merger_events.py:
    RegularGridInterpolator((z_grid, M_grid, fEdd_grid), pfEdd_given_Mz)

Typical use:
    python create_p_fEdd.py \
        --input-npz input/population_models/fEdd_M_z_dummy.npz \
        --output-npz input/population_models/pfEdd_given_Mz.npz

If the input fEdd field is constant 0.01 everywhere, the default behavior is:
    fEdd_grid = [0.01]
    pfEdd_given_Mz = 1 everywhere
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def infer_fedd_grid_from_field(fEdd_field: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Build a discrete fedd_grid from unique values found in the deterministic field.
    Values are clustered within tolerance by simple rounding logic.
    """
    vals = np.asarray(fEdd_field, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise ValueError("Input fEdd_grid field contains no finite values.")

    # Simple robust uniquing for floating-point values
    # scaled rounding avoids tiny numerical duplicates
    scale = max(1.0, np.nanmax(np.abs(vals)))
    decimals = max(0, int(np.ceil(-np.log10(tol / scale))))
    uniq = np.unique(np.round(vals, decimals=decimals).astype(float))

    if uniq.size == 0:
        raise ValueError("Could not infer any unique fEdd values from input field.")

    return np.sort(uniq)


def nearest_bin_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Return index of closest entry in grid for each value.
    """
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)

    # shape (..., Nf)
    dist = np.abs(values[..., None] - grid[None, None, :])
    return np.argmin(dist, axis=-1)


def build_pfedd_given_Mz(
    z_grid: np.ndarray,
    M_grid: np.ndarray,
    fEdd_grid: np.ndarray,
    p_input: np.ndarray,
    input_order: str = "zMf",
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and reshape an already-defined p(fEdd | M, z).

    Parameters
    ----------
    z_grid : array, shape (Nz,)
    M_grid : array, shape (NM,)
    fEdd_grid : array, shape (Nf,)
    p_input : array
        Probability table already defined on the (z, M, fEdd) grid,
        but possibly with a different axis order.
    input_order : str
        Axis order of p_input. Allowed:
            "zMf"  -> (Nz, NM, Nf)
            "zfM"  -> (Nz, Nf, NM)
            "Mzf"  -> (NM, Nz, Nf)
            "Mfz"  -> (NM, Nf, Nz)
            "fzM"  -> (Nf, Nz, NM)
            "fMz"  -> (Nf, NM, Nz)
    renormalize : bool
        If True, renormalize so that sum/integral over fEdd = 1 at each (z,M).

    Returns
    -------
    z_grid, M_grid, fEdd_grid, pfEdd_given_Mz
        with pfEdd_given_Mz shape (Nz, NM, Nf)
    """
    z_grid = np.asarray(z_grid, dtype=float)
    M_grid = np.asarray(M_grid, dtype=float)
    fEdd_grid = np.asarray(fEdd_grid, dtype=float)
    p_input = np.asarray(p_input, dtype=float)

    if z_grid.ndim != 1:
        raise ValueError("z_grid must be 1D.")
    if M_grid.ndim != 1:
        raise ValueError("M_grid must be 1D.")
    if fEdd_grid.ndim != 1:
        raise ValueError("fEdd_grid must be 1D.")

    Nz = z_grid.size
    NM = M_grid.size
    Nf = fEdd_grid.size

    expected_shapes = {
        "zMf": (Nz, NM, Nf),
        "zfM": (Nz, Nf, NM),
        "Mzf": (NM, Nz, Nf),
        "Mfz": (NM, Nf, Nz),
        "fzM": (Nf, Nz, NM),
        "fMz": (Nf, NM, Nz),
    }

    if input_order not in expected_shapes:
        raise ValueError(
            f"input_order={input_order!r} not recognized. "
            f"Allowed: {list(expected_shapes.keys())}"
        )

    if p_input.shape != expected_shapes[input_order]:
        raise ValueError(
            f"Input probability array has shape {p_input.shape}, "
            f"but for input_order='{input_order}' expected {expected_shapes[input_order]}."
        )

    # Reorder to (z, M, fEdd)
    transpose_map = {
        "zMf": (0, 1, 2),
        "zfM": (0, 2, 1),
        "Mzf": (1, 0, 2),
        "Mfz": (2, 0, 1),
        "fzM": (1, 2, 0),
        "fMz": (2, 1, 0),
    }
    pfEdd_given_Mz = np.transpose(p_input, axes=transpose_map[input_order])

    if np.any(~np.isfinite(pfEdd_given_Mz)):
        raise ValueError("Probability array contains non-finite values.")
    if np.any(pfEdd_given_Mz < 0):
        raise ValueError("Probability array contains negative values.")

    if renormalize:
        # discrete normalization over fEdd bins
        norm = pfEdd_given_Mz.sum(axis=2, keepdims=True)
        bad = (~np.isfinite(norm)) | (norm <= 0.0)
        if np.any(bad):
            raise ValueError("Some (z,M) cells have zero or invalid total probability over fEdd.")
        pfEdd_given_Mz = pfEdd_given_Mz / norm

    return z_grid, M_grid, fEdd_grid, pfEdd_given_Mz


def save_txt(outfile: str | Path,
             z_grid: np.ndarray,
             M_grid: np.ndarray,
             fEdd_grid: np.ndarray,
             pfEdd_given_Mz: np.ndarray) -> None:
    rows = []
    for i, z in enumerate(z_grid):
        for j, logM in enumerate(M_grid):
            rows.append(np.column_stack([
                np.full_like(fEdd_grid, z, dtype=float),
                np.full_like(fEdd_grid, logM, dtype=float),
                fEdd_grid,
                pfEdd_given_Mz[i, j, :]
            ]))
    arr = np.vstack(rows)
    header = (
        "z log10M fEdd pfEdd_given_Mz\n"
        "Columns:\n"
        "  z               redshift\n"
        "  log10M          log10(M/Msun)\n"
        "  fEdd            Eddington ratio bin center/value\n"
        "  pfedd_given_Mz  discrete conditional probability p(fEdd|M,z)"
    )
    np.savetxt(outfile, arr, header=header)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input-npz", required=True,
                   help="Input NPZ containing z_grid, M_grid, fEdd_grid (2D field).")
    p.add_argument("--output-npz", required=True,
                   help="Output NPZ for assign_redshift_to_merger_events.py")
    p.add_argument("--output-txt", default=None,
                   help="Optional TXT dump of the 3D probability table.")

    p.add_argument("--fEdd-grid", type=float, nargs="*", default=None,
                   help="Optional explicit discrete fEdd grid. "
                        "If omitted, infer unique values from the input field.")
    
    p.add_argument(
    "--input-order",
    default="zMf",
    choices=["zMf", "zfM", "Mzf", "Mfz", "fzM", "fMz"],
    help="Axis order of the probability array stored in the input file.")

    return p.parse_args()


def main():
    args = parse_args()

    d = np.load(args.input_npz, allow_pickle=True)

    required = {"z_grid", "M_grid", "fEdd_grid", "pfEdd_given_Mz"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(
            f"Missing required keys in input NPZ: {sorted(missing)}. "
            f"Available keys: {list(d.files)}"
        )

    z_grid = np.asarray(d["z_grid"], dtype=float)
    M_grid = np.asarray(d["M_grid"], dtype=float)
    fEdd_grid = np.asarray(d["fEdd_grid"], dtype=float)
    p_input = np.asarray(d["pfEdd_given_Mz"], dtype=float)

    z_grid, M_grid, fEdd_grid, pfEdd_given_Mz = build_pfedd_given_Mz(
        z_grid=z_grid,
        M_grid=M_grid,
        fEdd_grid=fEdd_grid,
        p_input=p_input,
        input_order=args.input_order,
        renormalize=True,
    )

    out_npz = Path(args.output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        z_grid=z_grid,
        M_grid=M_grid,
        fedd_grid=fEdd_grid,
        pfedd_given_Mz=pfEdd_given_Mz,
    )


if __name__ == "__main__":
    main()
