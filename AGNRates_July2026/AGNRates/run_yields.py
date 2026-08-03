#!/usr/bin/env python3
"""
For one simulation run directory, compute merger yields

    Y(t_merg | M, fEdd)

and the 2D histogram

    H(t_delay, Mchirp | M, fEdd)

separately for:
   - first_gen.txt
   - nth_generation.txt

Outputs are written in the same run directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# ---------------------- USER SETTINGS ------------------------
# ============================================================

# histogram bins 
T_BINS = np.logspace(-6, 4, 80) ### in Gyr
MCHIRP_BINS = np.logspace(1, 4, 80) ### in Msun

if np.any(np.diff(T_BINS) <= 0):
    raise ValueError("T_BINS must be strictly increasing")

if np.any(np.diff(MCHIRP_BINS) <= 0):
    raise ValueError("MCHIRP_BINS must be strictly increasing")

T_DELAY_COLUMN = "t_inspiral/Myr"
T_DELAY_TO_GYR = 1e-3

M1_COLUMN = "m1/Msun"

# True-merger selector and metadata
FLAG_COLUMN = "total_flags"


# ============================================================
# ---------------------- BASIC HELPERS ------------------------
# ============================================================

def read_events(file: Path) -> pd.DataFrame:
    return pd.read_csv(file, sep=r"\s+", comment="#", engine="python")

def parse(value):
    try:
        if not '_' in value:
            return int(value)
        raise ValueError
    except ValueError:
        try:
            if not '_' in value:
                return float(value)
            raise ValueError
        except ValueError:
            try:
                if ':' in value:
                    parts = value.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds, microseconds = map(float, parts[2].split('.')) if '.' in parts[2] else (int(parts[2]), 0)
                        return timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=int(microseconds))
                return datetime.strptime(value, "%y%m%d_%H%M")
            except ValueError:
                return value

def load_file(filename):
    params = {}
    data = {}
    
    with open(filename, 'r') as file:
        for line in file:
            if line=="Parameters:\n": continue
            elif line=="\n": continue
            elif line=="Data:\n": break
            else:
                line_splitted = line.strip().split()
                if len(line_splitted)==3: 
                    params[line_splitted[0]] = parse(line_splitted[2])
                else: 
                    params[line_splitted[0]] = []
                    for i in range(2, len(line_splitted)): 
                        params[line_splitted[0]].append(parse(line_splitted[i]))

        headers = file.readline().strip().split()
        data = {header: [] for header in headers}

        for line in file:
            values = line.strip().split()
            for i, header in enumerate(headers):
                data[header].append(parse(values[i]))
        
        for header in headers: data[header] = np.array(data[header])
    return params, data

def compute_chirp_mass(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Compute chirp mass in Msun:
        Mchirp = (m1 m2)^(3/5) / (m1 + m2)^(1/5)
    """
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)

    good = (
        np.isfinite(m1) & np.isfinite(m2) &
        (m1 > 0.0) & (m2 > 0.0) &
        np.isfinite(m1 + m2) & ((m1 + m2) > 0.0)
    )

    mchirp = np.full_like(m1, np.nan, dtype=float)
    mchirp[good] = (m1[good] * m2[good]) ** (3.0 / 5.0) / (m1[good] + m2[good]) ** (1.0 / 5.0)
    return mchirp


def compute_yield_histogram(
    t_delay_gyr: np.ndarray,
    Nsample: int,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        t_centers
        Y_bin    [per Gyr]

    normalized so that:
        sum(Y_bin * bin_width) = Nmerg / Nsample
    where:
        Nsample = total number of sampled rows/trials in the file
        Nmerg   = number of rows with an actual merger
    """
    t_delay_gyr = np.asarray(t_delay_gyr, float)
    t_delay_gyr = t_delay_gyr[np.isfinite(t_delay_gyr)]
    t_delay_gyr = t_delay_gyr[t_delay_gyr > 0.0]

    counts, edges = np.histogram(t_delay_gyr, bins=bins)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if Nsample <= 0:
        return centers, np.zeros_like(centers, dtype=float)

    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError(
            f"Non-positive or invalid bin widths in compute_yield_histogram: {widths}"
        )

    counts, _ = np.histogram(t_delay_gyr, bins=edges)
    Y_bin = counts / (Nsample * widths)
    return centers, Y_bin


def compute_2d_property_histogram(
    t_delay_gyr: np.ndarray,
    m1_msun: np.ndarray,
    Nsample: int,
    t_bins: np.ndarray,
    mchirp_bins: np.ndarray,
):
    t_delay_gyr = np.asarray(t_delay_gyr, float)
    m1_msun = np.asarray(m1_msun, float)

    good = (
        np.isfinite(t_delay_gyr)
        & np.isfinite(m1_msun)
        & (t_delay_gyr > 0.0)
        & (m1_msun > 0.0)
    )
    t_use = t_delay_gyr[good]
    m_use = m1_msun[good]

    t_edges = np.asarray(t_bins, dtype=float)
    m_edges = np.asarray(mchirp_bins, dtype=float)

    dt = np.diff(t_edges)
    dm = np.diff(m_edges)

    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("t_bins must be strictly increasing and finite")
    if np.any(~np.isfinite(dm)) or np.any(dm <= 0):
        raise ValueError("mchirp_bins must be strictly increasing and finite")

    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])

    if Nsample <= 0:
        H_2d = np.zeros((len(t_centers), len(m_centers)), dtype=float)
        return t_centers, m_centers, H_2d, 0

    counts, _, _ = np.histogram2d(
        t_use,
        m_use,
        bins=[t_edges, m_edges],
    )

    H_2d = counts / (Nsample * dt[:, None] * dm[None, :])
    return t_centers, m_centers, H_2d, int(len(t_use))


# ============================================================
# ---------------------- MAIN WORKFLOW ------------------------
# ============================================================

def process_event_file(
    event_file: Path,
    label: str,
    run_dir: Path,
):
    print(f"Processing {event_file}")

    params, data = load_file(event_file)
    df_events = pd.DataFrame(data)

    required_cols = [T_DELAY_COLUMN, FLAG_COLUMN, M1_COLUMN]
    missing = [col for col in required_cols if col not in df_events.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {event_file}: {missing}. "
            f"Available columns: {list(df_events.columns)}"
        )

    # Denominator = all sampled rows in the file
    Nsample = len(df_events)
    Nbh=params['N']

    # Select true mergers only
    is_merger = (df_events[FLAG_COLUMN] == 'in_cluster')
    df_mergers = df_events.loc[is_merger].copy()

    t_delay_gyr = (
        pd.to_numeric(df_mergers[T_DELAY_COLUMN], errors="coerce").to_numpy(dtype=float)
        * T_DELAY_TO_GYR
    )

    m1 = pd.to_numeric(df_mergers[M1_COLUMN], errors="coerce").to_numpy(dtype=float)

    # --------------------------------------------------------
    # 1D yield in t_delay
    # --------------------------------------------------------
    t_centers, Y = compute_yield_histogram(
        t_delay_gyr=t_delay_gyr,
        Nsample=Nsample,
        bins=T_BINS,
    )

    df_Y = pd.DataFrame({
        "t_delay_Gyr": t_centers,
        "Y_tdelay_given_M_fedd": Y,
    })
    df_Y.to_csv(
        run_dir / f"yield_{label}.txt",
        sep="\t",
        index=False,
    )

    widths = np.diff(T_BINS)
    yield_integral = float(np.sum(Y * widths))

    # --------------------------------------------------------
    # 2D histogram H(t_delay, Mchirp)
    # --------------------------------------------------------
    t2_centers, mch_centers, H_2d, _ = compute_2d_property_histogram(
        t_delay_gyr=t_delay_gyr,
        m1_msun=m1,
        Nsample=Nsample,
        t_bins=T_BINS,
        mchirp_bins=MCHIRP_BINS,
    )

    # Save as npz: compact and easy to load later
    np.savez_compressed(
        run_dir / f"hist2d_tdelay_mchirp_{label}.npz",
        H_tdelay_mchirp_given_M_fedd=H_2d,
        t_delay_edges_Gyr=T_BINS,
        mchirp_edges_Msun=MCHIRP_BINS,
        t_delay_centers_Gyr=t2_centers,
        mchirp_centers_Msun=mch_centers,
        normalization="counts / (Nsample * dt * dMchirp)",
        source_file=str(event_file),
        label=label,
    )

    dt = np.diff(T_BINS)
    dm = np.diff(MCHIRP_BINS)
    hist2d_integral = float(np.sum(H_2d * dt[:, None] * dm[None, :]))


    n_valid_hist2d = int(np.sum(
        np.isfinite(t_delay_gyr) &
        np.isfinite(m1) &
        (t_delay_gyr > 0.0) &
        (m1 > 0.0) &
        (t_delay_gyr >= T_BINS[0]) &
        (t_delay_gyr < T_BINS[-1]) &
        (m1 >= MCHIRP_BINS[0]) &
        (m1 < MCHIRP_BINS[-1])
    ))

    return {
        "file": str(event_file),
        "label": label,
        "Nsample": int(Nsample),
        "Nmerg": int(is_merger.sum()),
        "yield_integral": yield_integral,
        "hist2d_integral": hist2d_integral,
        "Nvalid_hist2d": n_valid_hist2d,
        "N_BH": Nbh,
        "yield_file": f"yield_{label}.txt",
        "hist2d_file": f"hist2d_tdelay_mchirp_{label}.npz",
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run-dir", required=True, help="Path to outputs/ directory")

    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    event_map = {
        "1g": run_dir / "first_gen.txt",
        "ng": run_dir / "nth_generation.txt",
    }

    summary_rows = []

    found_any = False
    for label, event_file in event_map.items():
        if not event_file.exists():
            print(f"[warning] missing file: {event_file}")
            continue

        found_any = True
        row = process_event_file(
            event_file=event_file,
            label=label,
            run_dir=run_dir,
        )
        summary_rows.append(row)

    if not found_any:
        raise FileNotFoundError(
            f"Neither first_gen.txt nor nth_generation.txt found in {run_dir}"
        )

    pd.DataFrame(summary_rows).to_csv(
        run_dir / "summary_yields.txt",
        sep="\t",
        index=False,
    )

    print(f"Done. Outputs written in {run_dir}")


if __name__ == "__main__":
    main()
