from pathlib import Path

import numpy as np
import pandas as pd
import argparse

from datetime import datetime, timedelta

MERGER_FLAG = 7
DELAY_COLUMN = "t_inspiral/Myr"
FLAG_COLUMN = "total_flags"
BIN_EDGES_GYR = np.logspace(-9, -1, 80)

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

def make_yield_file(LABEL, 
    input_file,
):
    """Create the two-column LISA delay-time yield file."""
    params={}
    with input_file.open() as file:
        for line in file:
            if line.strip() == "Data:":
                break
            if line=="Parameters:\n": continue
            elif line=="\n": continue
            else:
                line_splitted = line.strip().split()
                if len(line_splitted)==3: 
                    params[line_splitted[0]] = parse(line_splitted[2])
                else: 
                    params[line_splitted[0]] = []
                    for i in range(2, len(line_splitted)): 
                        params[line_splitted[0]].append(parse(line_splitted[i]))
            
        events = pd.read_csv(file, sep=r"\s+", engine="python")

    print(f'events: {events}')

    N_SAMPLE=params['N']
    N_BH=params['N']

    columns = {column.rstrip(","): column for column in events.columns}

    missing = {DELAY_COLUMN, FLAG_COLUMN} - columns.keys()
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if N_SAMPLE <= 0:
        raise ValueError("N_SAMPLE must be positive")

    is_merger = events[columns[FLAG_COLUMN]] == MERGER_FLAG
    delays_gyr = (
        pd.to_numeric(events.loc[is_merger, columns[DELAY_COLUMN]], errors="coerce")
        .to_numpy(dtype=float)
        * 1e-3
    )
    delays_gyr = delays_gyr[np.isfinite(delays_gyr) & (delays_gyr > 0.0)]

    bin_edges = BIN_EDGES_GYR.copy()
    if len(delays_gyr) > 0 and delays_gyr.min() < bin_edges[0]:
        lower_edge = 10.0 ** np.floor(np.log10(delays_gyr.min()))
        bin_edges = np.logspace(
            np.log10(lower_edge),
            np.log10(BIN_EDGES_GYR[-1]),
            len(BIN_EDGES_GYR),
        )

    counts, _ = np.histogram(delays_gyr, bins=bin_edges)
    widths = np.diff(bin_edges)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    yields = counts / (N_SAMPLE * widths)

    if np.any(~np.isfinite(yields)) or np.any(yields < 0.0):
        raise ValueError("Generated yields must be finite and non-negative")
    
    M=params['M_SMBH']
    alpha=params['alpha']

    output_file=Path(f'/Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/{LABEL}/M_{M:.1e}_fEdd_0.01/yield_1g.txt')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "t_delay_Gyr": centers,
            "Y_dt_delay_Gyr_given_M_fedd": yields,
        }
    ).to_csv(output_file, sep="\t", index=False)

    pd.DataFrame({"label": [LABEL], "N_BH": [N_BH]}).to_csv(
        output_file.parent / "summary_yields.txt",
        sep="\t",
        index=False,
    )

    return {
        "file": str(output_file),
        "summary_file": str(output_file.parent / "summary_yields.txt"),
        "label": LABEL,
        "Nsample": N_SAMPLE,
        "Nmerg": int(is_merger.sum()),
        "N_BH": N_BH,
        "yield_integral": float(np.sum(yields * widths)),
    }

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--infile", required=True, help="input file")
    p.add_argument("--label", required=True, help="Physical-model directory below SG_alpha_<alpha>")

    return p.parse_args()

def main():
    args = parse_args()
    make_yield_file(LABEL=args.label, input_file=Path(args.infile))

if __name__ == "__main__":
    main()