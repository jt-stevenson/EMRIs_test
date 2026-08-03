import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_mchirp_percentiles_file(filename):
    df = pd.read_csv(filename, sep=r"\s+|\t+", engine="python", comment="#")

    required = {
        "z",
        "mchirp_p5p0_Msun",
        "mchirp_p50p0_Msun",
        "mchirp_p95p0_Msun",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{filename} is missing columns: {sorted(missing)}")

    return {
        "z": pd.to_numeric(df["z"], errors="coerce").to_numpy(dtype=float),
        "p5": pd.to_numeric(df["mchirp_p5p0_Msun"], errors="coerce").to_numpy(dtype=float),
        "p50": pd.to_numeric(df["mchirp_p50p0_Msun"], errors="coerce").to_numpy(dtype=float),
        "p95": pd.to_numeric(df["mchirp_p95p0_Msun"], errors="coerce").to_numpy(dtype=float),
    }


def rebin_xy(z, y, bin_edges, statistic="mean"):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(z) & np.isfinite(y)
    z = z[good]
    y = y[good]

    inds = np.digitize(z, bin_edges) - 1
    nbins = len(bin_edges) - 1

    z_new = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_new = np.full(nbins, np.nan)

    for i in range(nbins):
        m = inds == i
        if np.any(m):
            if statistic == "mean":
                y_new[i] = np.mean(y[m])
            elif statistic == "median":
                y_new[i] = np.median(y[m])
            elif statistic == "sum":
                y_new[i] = np.sum(y[m])
            else:
                raise ValueError("statistic must be 'mean', 'median', or 'sum'")

    return z_new, y_new


def prepare_percentile_curve(out, bin_edges):
    z = np.asarray(out["z"], dtype=float)
    p5 = np.asarray(out["p5"], dtype=float)
    p50 = np.asarray(out["p50"], dtype=float)
    p95 = np.asarray(out["p95"], dtype=float)

    good = (
        np.isfinite(z)
        & np.isfinite(p5) & (p5 > 0)
        & np.isfinite(p50) & (p50 > 0)
        & np.isfinite(p95) & (p95 > 0)
    )

    z_to_be_binned = z[good]
    p5 = p5[good]
    p50 = p50[good]
    p95 = p95[good]

    z, p5 = rebin_xy(z_to_be_binned, p5, bin_edges, statistic="mean")
    _, p50 = rebin_xy(z_to_be_binned, p50, bin_edges, statistic="mean")
    _, p95 = rebin_xy(z_to_be_binned, p95, bin_edges, statistic="mean")

    good = (
        np.isfinite(z)
        & np.isfinite(p5) & (p5 > 0)
        & np.isfinite(p50) & (p50 > 0)
        & np.isfinite(p95) & (p95 > 0)
    )
    return z[good], p5[good], p50[good], p95[good]


def z_to_lookback(z):
    return cosmo.lookback_time(z).value  # Gyr


# build interpolation grid once
z_grid = np.linspace(0, 11, 1000)
t_grid = cosmo.lookback_time(z_grid).value
t_to_z = interp1d(t_grid, z_grid, bounds_error=False, fill_value="extrapolate")


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

alphas = ["0.01", "0.1"]
progrs = ["all_prograde", "all_retrograde", "agnostic"]
redshifts = ["EL_LAM", "EL_HAM", "SE_HAM"]

base_dir = "../outputs/MRD_results"
base_models = ["G24_K18-3bb_0.0-IG25"]

bin_edges = np.arange(0.15, 10.6, 0.3)

spectral = cm.get_cmap("coolwarm_r")
palette = {
    "0.01": {  # low-alpha, warm side
        "all_retrograde": spectral(0.30),
        "agnostic": spectral(0.15),
        "all_prograde": spectral(0.00),
    },
    "0.1": {   # high-alpha, cool side
        "all_retrograde": spectral(0.70),
        "agnostic": spectral(0.85),
        "all_prograde": spectral(1.00),
    },
}

markers = {
    "0.01": {"EL": "o", "SE": "s"},
    "0.1": {"EL": "^", "SE": "D"},
}

linestyle = {
    "EL_LAM": "-",
    "EL_HAM": (0, (3, 2)),
    "SE_HAM": (2.5, (1, 2)),
}


# --------------------------------------------------
# LOAD FILES + PLOT
# --------------------------------------------------

os.makedirs("../plots", exist_ok=True)

for base_model in base_models:
    mchirp_outputs = {}

    # ----------------------------
    # load all progr files
    # ----------------------------
    for alpha in alphas:
        for progr in progrs:
            run_label = f"{base_model}-{progr}-tau_x_1."

            for redshift in redshifts:
                folder = os.path.join(base_dir, f"alpha_{alpha}", run_label, redshift)

                if not os.path.isdir(folder):
                    print(f"Skipping missing folder {folder}")
                    continue

                files = sorted(glob.glob(os.path.join(folder, "mchirp_percentiles_vs_z*.txt")))
                if len(files) == 0:
                    print(f"No Mchirp percentile file found in {folder}")
                    continue

                if len(files) > 1:
                    print(f"Multiple Mchirp percentile files found in {folder}, using {files[0]}")

                file = files[0]
                mchirp_outputs[(alpha, progr, redshift)] = read_mchirp_percentiles_file(file)

    print("Loaded keys:")
    print(mchirp_outputs.keys())

    # ----------------------------
    # plot
    # ----------------------------
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(3,1,figsize=(10, 21))

    for alpha in alphas:
        for redshift in redshifts[::-1]:
            ax=axs[redshifts.index(redshift)]
            pop, abundance = redshift.split("_")

            keys = [(alpha, progr, redshift) for progr in progrs]
            missing = [k for k in keys if k not in mchirp_outputs]
            keys = [(alpha, progr, redshift) for progr in progrs if (alpha, progr, redshift) in mchirp_outputs]

            if len(keys) == 0:
                print(f"No available entries for alpha={alpha}, redshift={redshift}")
                continue
            
            if len(keys) < len(progrs):
                print(f"Using only available progr entries for alpha={alpha}, redshift={redshift}: {keys}")

            z_list = []
            p5_list = []
            p50_list = []
            p95_list = []

            for k in keys:
                z_i, p5_i, p50_i, p95_i = prepare_percentile_curve(mchirp_outputs[k], bin_edges)
                z_list.append(z_i)
                p5_list.append(p5_i)
                p50_list.append(p50_i)
                p95_list.append(p95_i)

            z_ref = z_list[0]
            same_grid = all(len(z_ref) == len(z_i) and np.allclose(z_ref, z_i) for z_i in z_list[1:])
            if not same_grid:
                print(f"Inconsistent z grids for alpha={alpha}, redshift={redshift}")
                continue

            p5_stack = np.vstack(p5_list)
            p50_stack = np.vstack(p50_list)
            p95_stack = np.vstack(p95_list)

            # Envelope over prograde assumptions, following the same logic
            p_low = np.nanmin(p5_stack, axis=0)
            p_high = np.nanmax(p95_stack, axis=0)
            p_mid = np.nanmean(p50_stack, axis=0)

            color_main = palette[alpha]["agnostic"]
            color_fill = palette[alpha]["all_prograde"]

            marker = markers[alpha][pop]
            ls = linestyle[redshift]
            mfc = "white" if abundance == "HAM" else color_main

            good_fill = (
                np.isfinite(z_ref)
                & np.isfinite(p_low)
                & np.isfinite(p_high)
                & (p_low > 0)
                & (p_high > 0)
            )

            if np.any(good_fill):
                ax.fill_between(
                    z_ref[good_fill],
                    p_low[good_fill],
                    p_high[good_fill],
                    color=color_fill,
                    alpha=0.2,
                    linewidth=0,
                )

            good_line = np.isfinite(z_ref) & np.isfinite(p_mid) & (p_mid > 0)

            if np.any(good_line):
                ax.plot(
                    z_ref[good_line],
                    p_mid[good_line],
                    marker=marker,
                    linestyle=ls,
                    color=color_main,
                    markerfacecolor=mfc,
                    markeredgecolor=color_main,
                    markersize=5.0,
                    linewidth=2.0,
                    markevery=1,
                    label=f"alpha_{alpha}_{redshift}",
                )

    axs[0].set_xlabel("z")
    axs[0].set_ylabel(r"$\mathcal{M}_{\rm chirp}$ [$M_\odot$]")
    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlim(0, 10.1)

    ax_top = axs[2].secondary_xaxis('top', functions=(z_to_lookback, t_to_z))
    ax_top.set_xlabel("Lookback time [Gyr]")
    ax_top.tick_params(axis='both')
    ax_top.set_xticks([0, 4, 8, 10, 12, 13])


    legend_handles = [
        Line2D([0], [0], color=palette["0.01"]["all_prograde"], lw=2, label=r"$\alpha = 0.01$"),
        Line2D([0], [0], color=palette["0.1"]["all_prograde"], lw=2, label=r"$\alpha = 0.1$"),
        Line2D([0], [0], color="gold", lw=1.5, label=r"50th percentile"),
        plt.Rectangle((0, 0), 1, 1, facecolor="gold", alpha=0.2, edgecolor="none", label=r"5$-$95 percentile"),
        Line2D([0], [0], color="darkgray", marker="o", linestyle="None", markersize=6, label="EL, " + r"$\alpha=0.01$"),
        Line2D([0], [0], color="darkgray", marker="^", linestyle="None", markersize=6, label="EL, " + r"$\alpha=0.1$"),
        Line2D([0], [0], color="darkgray", marker="s", linestyle="None", markersize=6, label="SE, " + r"$\alpha=0.01$"),
        Line2D([0], [0], color="darkgray", marker="D", linestyle="None", markersize=6, label="SE, " + r"$\alpha=0.1$"),
        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["EL_LAM"], label="fiducial"),
        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["EL_HAM"], label="CAT-EL"),
        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["SE_HAM"], label="CAT-SE"),
    ]

    axs[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fontsize=11,
        ncol=2,
    )

    plt.savefig(f"../plots/Mchirp_vs_z_{base_model}_figure.pdf", bbox_inches="tight")
    print(f"Saved plot for {base_model}")
    plt.show()
