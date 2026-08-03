import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

### MUST conda activate o4b-astro 


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_mrd_file(filename):
    data = np.genfromtxt(filename, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    return {
        "z_mid": data[:, 0],
        "R_source": data[:, 1],
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


def prepare_curve(out, bin_edges):
    z = np.asarray(out["z_mid"], dtype=float)
    R = np.asarray(out["R_source"], dtype=float)

    good = np.isfinite(z) & np.isfinite(R) & (R > 0)
    z = z[good]
    R = R[good]

    z, R = rebin_xy(z, R, bin_edges, statistic="mean")

    good = np.isfinite(z) & np.isfinite(R) & (R > 0)
    z = z[good]
    R = R[good]

    # extrapolate to z=0 in log-space using first two valid points
    if len(z) > 1 and z[0] > 0 and R[0] > 0 and R[1] > 0:
        logR0 = np.log10(R[0])
        logR1 = np.log10(R[1])
        slope_log = (logR1 - logR0) / (z[1] - z[0])
        logR_at_0 = logR0 - slope_log * z[0]
        R_at_0 = 10**logR_at_0

        z = np.insert(z, 0, 0.0)
        R = np.insert(R, 0, R_at_0)

    return z, R

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
progrs = ["agnostic"]#, "all_prograde", "all_retrograde"]
redshifts = ["EL_LAM", "EL_HAM", "SE_HAM"]

base_dir = "outputs/MRD_results"
base_models = [
    "G24_K18-3bb_0.0-IG25",
    "G24_K18-3bb_0.0-IG20", 
    "G24_K18-3bb_0.0-Calcino23", 
    "G24_K18-3bb_0.1-IG25",
    "B16_K18-3bb_0.0-IG25"
    ]

bin_edges = np.arange(0.15, 10.6, 0.3)

spectral = cm.get_cmap("PRGn") 
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
# LVK
# --------------------------------------------------
from popsummary.popresult import PopulationResult

strong_bbh_O4b = PopulationResult(fname=f'outputs/MRD_results/LVK/GWTC5/gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5')
z_pixelpop = PopulationResult(fname=f'outputs/MRD_results/LVK/GWTC5/z_varcut1_popsummary.h5')

strong_bbh_O4b_z_grid, strong_bbh_O4b_Rz = strong_bbh_O4b.get_rates_on_grids('redshift')
strong_bbh_O4b_z_grid = strong_bbh_O4b_z_grid.reshape((strong_bbh_O4b_z_grid.size, ))
pixelpop_z_grid, pixelpop_z = z_pixelpop.get_rates_on_grids('redshift')
pixelpop_z_grid = pixelpop_z_grid.reshape((pixelpop_z_grid.size, ))


##)

from scipy.ndimage import gaussian_filter1d

def smooth_positive_curve(y, sigma=1.0):
    """
    Smooth a positive curve in log-space for plotting.

    Parameters
    ----------
    y : array
        Positive-valued curve.
    sigma : float
        Gaussian smoothing width in index units.
        Try sigma=1.0--2.0.
    """

    y = np.asarray(y, dtype=float)

    good = np.isfinite(y) & (y > 0)
    y_smooth = np.full_like(y, np.nan, dtype=float)

    if good.sum() < 3:
        return y

    logy = np.log10(y[good])
    logy_smooth = gaussian_filter1d(logy, sigma=sigma, mode="nearest")

    y_smooth[good] = 10.0**logy_smooth

    return y_smooth

# --------------------------------------------------
# LOAD FILES + PLOT
# --------------------------------------------------

os.makedirs("plots", exist_ok=True)

for base_model in base_models:
    mrd_outputs = {}

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

                files = sorted(glob.glob(os.path.join(folder, "MRD_vs_z*.txt")))
                if len(files) == 0:
                    print(f"No MRD file found in {folder}")
                    continue

                if len(files) > 1:
                    print(f"Multiple MRD files found in {folder}, using {files[0]}")

                file = files[0]
                mrd_outputs[(alpha, progr, redshift)] = read_mrd_file(file)

    print("Loaded keys:")
    print(mrd_outputs.keys())

    # ---------------------------------------------------------------------------------------------------------------------------
    # plot, FIGURE 1
    # ---------------------------------------------------------------------------------------------------------------------------
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(10, 7))

    ## LVK, GWTC5:
    spm_color = 'black'
    wpm_color = 'darkorange'

    ax.step(pixelpop_z_grid, np.quantile(pixelpop_z, 0.5, axis = 0),  lw=1.2, color=wpm_color, label='$\\textsc{PixelPop, gwtc-5.0}$', zorder=1)
    ax.step(pixelpop_z_grid, np.quantile(pixelpop_z, 0.05, axis = 0), lw=0.8, color=wpm_color, label='$\\textsc{PixelPop, gwtc-5.0}$', zorder=1)
    ax.step(pixelpop_z_grid, np.quantile(pixelpop_z, 0.95, axis = 0), lw=0.8, color=wpm_color, label='$\\textsc{PixelPop, gwtc-5.0}$', zorder=1)
    ax.fill_between(pixelpop_z_grid, np.quantile(pixelpop_z, 0.05, axis = 0), 
                 np.quantile(pixelpop_z, 0.95, axis = 0), 
                 color=wpm_color, edgecolor=None, alpha=0.1, step='pre', zorder=0)

    ax.semilogy(strong_bbh_O4b_z_grid, np.quantile(strong_bbh_O4b_Rz, 0.5, axis = 0),  lw=1.2, color=spm_color, label='$\\textsc{DefaultBBH, gwtc-5.0}$', zorder=1)
    ax.semilogy(strong_bbh_O4b_z_grid, np.quantile(strong_bbh_O4b_Rz, 0.05, axis = 0), lw=0.8, color=spm_color, label='$\\textsc{DefaultBBH, gwtc-5.0}$', zorder=1)
    ax.semilogy(strong_bbh_O4b_z_grid, np.quantile(strong_bbh_O4b_Rz, 0.95, axis = 0), lw=0.8, color=spm_color, label='$\\textsc{DefaultBBH, gwtc-5.0}$', zorder=1)
    ax.fill_between(strong_bbh_O4b_z_grid, np.quantile(strong_bbh_O4b_Rz, 0.05, axis = 0), 
                 np.quantile(strong_bbh_O4b_Rz, 0.95, axis = 0), 
                 color=spm_color, edgecolor=None, alpha=0.1, zorder=0)
    
    #ax.semilogy(strong_bbh_O4b_z_grid, 0.1 * np.quantile(strong_bbh_O4b_Rz, 0.5, axis = 0), 
    #         color=spm_color, ls=":", label='$\\textsc{Default BBH, gwtc-5.0}$', zorder=0)


    for alpha in alphas:
        for redshift in redshifts[::-1]:
            pop, abundance = redshift.split("_")

            keys = [(alpha, progr, redshift) for progr in progrs]
            missing = [k for k in keys if k not in mrd_outputs]
            if missing:
                print(f"Missing progr entries for alpha={alpha}, redshift={redshift}: {missing}")
                continue

            z_list = []
            R_list = []

            for k in keys:
                z_i, R_i = prepare_curve(mrd_outputs[k], bin_edges)
                z_list.append(z_i)
                R_list.append(R_i)

            z_ref = z_list[0]
            if not all(len(z_ref) == len(z_i) and np.allclose(z_ref, z_i) for z_i in z_list[1:]):
                print(f"Inconsistent z grids for alpha={alpha}, redshift={redshift}")
                continue

            curves = np.vstack(R_list)

            R_low = np.nanmin(curves, axis=0)
            R_high = np.nanmax(curves, axis=0)
            R_avg = np.nanmean(curves, axis=0)

            # Smooth only for plotting, not for saved data.
            SMOOTH_SIGMA = 1.5  # try 0.8--2.0
            R_low  = smooth_positive_curve(R_low,  sigma=SMOOTH_SIGMA)
            R_high = smooth_positive_curve(R_high, sigma=SMOOTH_SIGMA)
            R_avg  = smooth_positive_curve(R_avg,  sigma=SMOOTH_SIGMA)

            # style choice:
            # dark line = agnostic color
            # fill = prograde color
            color_main = palette[alpha]["agnostic"]
            color_fill = palette[alpha]["all_prograde"]

            marker = markers[alpha][pop]
            ls = linestyle[redshift]
            mfc = "white" if abundance == "HAM" else color_main

            good_fill = (
                np.isfinite(z_ref) &
                np.isfinite(R_low) &
                np.isfinite(R_high) &
                (R_low > 0) &
                (R_high > 0)
            )

            if np.any(good_fill):
                ax.fill_between(
                    z_ref[good_fill],
                    R_low[good_fill],
                    R_high[good_fill],
                    color=color_fill,
                    alpha=0.2,
                    linewidth=0,
                )

            good_line = (
                np.isfinite(z_ref) &
                np.isfinite(R_avg) &
                (R_avg > 0)
            )

            if np.any(good_line):
                ax.plot(
                    z_ref[good_line],
                    R_avg[good_line],
                    marker=None,#marker,
                    linestyle=ls,
                    color=color_main,
                    markerfacecolor=mfc,
                    markeredgecolor=color_main,
                    markersize=5.,
                    linewidth=2.,
                    markevery=1,
                    label=f"alpha_{alpha}_{redshift}",
                )


    ax.set_xlabel("z")
    ax.set_ylabel(r"$\mathcal{R}(z)$ [Gpc$^{-3}$ yr$^{-1}$]")
    ax.set_yscale("log")
    #ax.set_xscale("log")
    #ax.set_xlim(5e-1,1.1e1)
    ax.set_xlim(0,9.5)
    ax.set_ylim(5e-2,5e4)
    
    ax_top = ax.secondary_xaxis('top', functions=(z_to_lookback, t_to_z))
    ax_top.set_xlabel("Lookback time [Gyr]")
    ax_top.tick_params(axis='both')
    ax_top.set_xticks([0,4,8,10,12,13])

    
    # vertical colored regions
    #ax.axvspan(0, 1.7, color='#e3d7f3', alpha=0.2)
    #ax.axvspan(1.7, 4, color='#f0f0f0', alpha=0.2)
    #ax.axvspan(4,  10, color='#f3e6c9', alpha=0.2)
    # labels
    ax.annotate("", xy=(1.7, 0.965), xytext=(0, 0.965), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#5e4b8a"), annotation_clip=False)
    ax.text(0.85, 0.95, "AGN population is\n data-constrained\n (Shen+ 2020)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#5e4b8a", clip_on=False)
    ax.annotate("", xy=(4, 0.965), xytext=(1.7, 0.965), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#4d4d4d"), annotation_clip=False)
    ax.text(2.85, 0.95, "extrapolation", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#4d4d4d", clip_on=False)
    ax.annotate("", xy=(9.5, 0.965), xytext=(4, 0.965), xycoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#8a6b3d"), annotation_clip=False)
    ax.text(7., 0.95, "AGN population is simulation-driven (CAT)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#8a6b3d", clip_on=False)
    
    # legend
    EL_handle=(
        Line2D([0], [0], color="darkgray", marker="o", linestyle="None", markersize=6),
        Line2D([0], [0], color="darkgray", marker="^", linestyle="None", markersize=6, label="EL"),
    )
    SE_handle=(
        Line2D([0], [0], color="darkgray", marker="s", linestyle="None", markersize=6),
        Line2D([0], [0], color="darkgray", marker="D", linestyle="None", markersize=6, label="SE"),
    )
    legend_handles = [
    
        Line2D([0], [0], color=palette["0.01"]["agnostic"], lw=2, label=r"$\alpha = 0.01$"),
        Line2D([0], [0], color=palette["0.1"]["agnostic"],  lw=2, label=r"$\alpha = 0.1$"),

        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["EL_LAM"], label="fiducial"),
        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["EL_HAM"], label="CAT-EL"),
        Line2D([0], [0], color="darkgray", lw=1.8, linestyle=linestyle["SE_HAM"], label="CAT-SE"),

        
        Patch(facecolor=spm_color, edgecolor=spm_color, alpha=0.5, label="GWTC5"),
        Patch(facecolor=wpm_color, edgecolor=wpm_color, alpha=0.5, label="GWTC5"),

        Line2D([0], [0], color="darkgray", lw=1.5, label="average over models"),
        Patch(facecolor="darkgray", alpha=0.2, edgecolor="none", label=r"min$-$max over models"),
        ]

    legend_labels = [
        r"$\alpha = 0.01$",
        r"$\alpha = 0.1$",
        "fiducial",
        "CAT-EL",
        "CAT-SE",
        "GWTC-5, DefaultBBH",
        "GWTC-5, PixelPop",
        "average over models",
        r"$90\%$ enclosed",
    ]

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower left",
        frameon=True,
        fontsize=11,
        ncol=2,
    )



    plt.savefig(f"plots/MRD_{base_model}_figure.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot for {base_model}")
    plt.show()



