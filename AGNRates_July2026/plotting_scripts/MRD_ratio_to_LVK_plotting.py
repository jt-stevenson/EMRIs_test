import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d

### MUST conda activate 04b-astro


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


def smooth_positive_curve(y, sigma=1.0):
    """
    Smooth a positive curve in log-space for plotting / summary consistency.
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


def make_lvk_median_interpolator(z_lvk, R_lvk):
    """
    Log-space interpolator for the LVK Default BBH median rate.
    Returns NaN outside the valid LVK redshift range.
    """
    z_lvk = np.asarray(z_lvk, dtype=float)
    R_lvk = np.asarray(R_lvk, dtype=float)

    good = np.isfinite(z_lvk) & np.isfinite(R_lvk) & (R_lvk > 0)
    z_lvk = z_lvk[good]
    R_lvk = R_lvk[good]

    order = np.argsort(z_lvk)
    z_lvk = z_lvk[order]
    R_lvk = R_lvk[order]

    log_interp = interp1d(
        z_lvk,
        np.log10(R_lvk),
        bounds_error=False,
        fill_value=np.nan,
    )

    def R_lvk_default_median(z):
        logR = log_interp(z)
        return 10.0**logR

    return R_lvk_default_median


def integrated_fraction_against_lvk(z, R, R_lvk_func, z_min, z_max):
    """
    Compute int R_AGN dz / int R_LVK dz over the LVK Default BBH redshift support.

    This is an integrated MRD fraction, not a detected/event fraction.
    """
    z = np.asarray(z, dtype=float)
    R = np.asarray(R, dtype=float)

    good = (
        np.isfinite(z)
        & np.isfinite(R)
        & (R > 0)
        & (z >= z_min)
        & (z <= z_max)
    )

    z = z[good]
    R = R[good]

    if len(z) < 2:
        return np.nan

    order = np.argsort(z)
    z = z[order]
    R = R[order]

    R_lvk = R_lvk_func(z)

    good = np.isfinite(R_lvk) & (R_lvk > 0)
    z = z[good]
    R = R[good]
    R_lvk = R_lvk[good]

    if len(z) < 2:
        return np.nan

    num = np.trapz(R, z)
    den = np.trapz(R_lvk, z)

    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan

    return num / den


# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

alphas = ["0.01", "0.1"]
progrs = ["agnostic", "all_prograde", "all_retrograde"]
redshifts = ["EL_LAM", "EL_HAM", "SE_HAM"]

base_dir = "outputs/MRD_results"
base_models = [
    "G24_K18-3bb_0.0-IG25",
    "G24_K18-3bb_0.0-IG20",
    "G24_K18-3bb_0.0-Calcino23",
    "G24_K18-3bb_0.1-IG25",
    "B16_K18-3bb_0.0-IG25",
]

bin_edges = np.arange(0.15, 10.6, 0.3)

spectral = cm.get_cmap("PRGn")
palette = {
    "0.01": {
        "all_retrograde": spectral(0.30),
        "agnostic": spectral(0.15),
        "all_prograde": spectral(0.00),
    },
    "0.1": {
        "all_retrograde": spectral(0.70),
        "agnostic": spectral(0.85),
        "all_prograde": spectral(1.00),
    },
}


# --------------------------------------------------
# LVK
# --------------------------------------------------

from popsummary.popresult import PopulationResult

strong_bbh_O4b = PopulationResult(
    fname=(
        "outputs/MRD_results/LVK/GWTC5/"
        "gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_"
        "redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_"
        "popsummary_result.h5"
    )
)

strong_bbh_O4b_z_grid, strong_bbh_O4b_Rz = strong_bbh_O4b.get_rates_on_grids("redshift")
strong_bbh_O4b_z_grid = strong_bbh_O4b_z_grid.reshape((strong_bbh_O4b_z_grid.size,))

# LVK Default BBH denominator: use the median LVK curve.
strong_med = np.quantile(strong_bbh_O4b_Rz, 0.50, axis=0)
strong_low = np.quantile(strong_bbh_O4b_Rz, 0.05, axis=0)
strong_high = np.quantile(strong_bbh_O4b_Rz, 0.95, axis=0)

lvk_good = (
    np.isfinite(strong_bbh_O4b_z_grid)
    & np.isfinite(strong_med)
    & np.isfinite(strong_low)
    & np.isfinite(strong_high)
    & (strong_med > 0)
    & (strong_low > 0)
    & (strong_high > 0)
)

z_lvk = strong_bbh_O4b_z_grid[lvk_good]
R_lvk_med = strong_med[lvk_good]

z_lvk_min = np.nanmin(z_lvk)
z_lvk_max = np.nanmax(z_lvk)

R_lvk_default_median = make_lvk_median_interpolator(z_lvk, R_lvk_med)

print(f"Using LVK Default BBH redshift range: z = {z_lvk_min:.3f} -- {z_lvk_max:.3f}")


# --------------------------------------------------
# LOAD FILES + PLOT
# --------------------------------------------------

os.makedirs("plots", exist_ok=True)

for base_model in base_models:
    mrd_outputs = {}

    # ----------------------------
    # Load all files
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

    # --------------------------------------------------
    # Collect fractions by alpha only
    # --------------------------------------------------
    fractions_by_alpha = {alpha: [] for alpha in alphas}

    for alpha in alphas:
        for redshift in redshifts:
            for progr in progrs:
                key = (alpha, progr, redshift)

                if key not in mrd_outputs:
                    print(f"Missing {key}")
                    continue

                z_i, R_i = prepare_curve(mrd_outputs[key], bin_edges)

                # Smooth only if you want the same plotting-level treatment
                # as in the MRD curves. For a more literal integral, comment this out.
                R_i = smooth_positive_curve(R_i, sigma=1.5)

                f_i = integrated_fraction_against_lvk(
                    z_i,
                    R_i,
                    R_lvk_func=R_lvk_default_median,
                    z_min=z_lvk_min,
                    z_max=z_lvk_max,
                )

                print(f"zmin={z_lvk_min}, zmax={z_lvk_max}, f_i={f_i}")

                if np.isfinite(f_i) and f_i > 0:
                    fractions_by_alpha[alpha].append(f_i)

    for alpha in alphas:
        fractions_by_alpha[alpha] = np.asarray(fractions_by_alpha[alpha], dtype=float)
        print(alpha, fractions_by_alpha[alpha])

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------
    summary = {}

    for alpha in alphas:
        f = fractions_by_alpha[alpha]
        f = f[np.isfinite(f) & (f > 0)]

        if len(f) == 0:
            summary[alpha] = None
            continue

        summary[alpha] = {
            "mean": np.mean(f),
            "p00p05": np.percentile(f, 0.05),
            "p16": np.percentile(f, 16),
            "p84": np.percentile(f, 84),
            "p99p05": np.percentile(f, 99.05),
            "n": len(f),
        }

        print(
            alpha,
            "N =", summary[alpha]["n"],
            "mean =", summary[alpha]["mean"],
            "68% =", (summary[alpha]["p16"], summary[alpha]["p84"]),
            "99% =", (summary[alpha]["p00p05"], summary[alpha]["p99p05"]),
        )

    # --------------------------------------------------
    # Plot: one point per alpha, with 68% and 99% intervals
    # --------------------------------------------------
    plt.rcParams.update({"font.size": 12})

    fig, ax = plt.subplots(figsize=(5, 4))
    x_positions = np.arange(len(alphas))

    for i, alpha in enumerate(alphas):
        s = summary[alpha]

        if s is None:
            continue

        mean = s["mean"]
        color = palette[alpha]["agnostic"]

        # 68% interval
        yerr_68 = np.array([
            [mean - s["p16"]],
            [s["p84"] - mean],
        ])

        # 99% interval
        yerr_99 = np.array([
            [mean - s["p00p05"]],
            [s["p99p05"] - mean],
        ])

        # 99% interval: thin, faint
        ax.errorbar(
            x_positions[i],
            mean,
            yerr=yerr_99,
            fmt="none",
            color=color,
            elinewidth=1.3,
            capsize=6,
            alpha=0.7,
            zorder=19,
        )

        # 68% interval: thick, darker
        ax.errorbar(
            x_positions[i],
            mean,
            yerr=yerr_68,
            fmt="none",
            color=color,
            elinewidth=3.0,
            capsize=0,
            alpha=0.9,
            zorder=20,
        )

        # Mean point
        ax.scatter(
            x_positions[i],
            mean,
            s=90,
            color=color,
            edgecolor=None,
            linewidth=0.8,
            zorder=21,
            label=rf"$\alpha={alpha}$",
        )


    # Reference levels
    ax.axhline( 1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.8, label=r"$R_{\rm AGN}=R_{\rm LVK,DefaultBBH}$", )

    ymin, ymax = ax.get_ylim()
    xarrow = {0.1: 0.2, 0.15: -0.05, 0.21: -0.3}
    arrow_label = {0.1: "bright AGNs", 0.15: "AGN flares", 0.21: "faint AGNs"}
    for f_ref in [0.1, 0.15, 0.21]:
        ax.axhline( f_ref, color="gray", linestyle="-", linewidth=1., zorder=0)
        ax.axhspan( ymin, f_ref, color="gray", alpha=0.1, linewidth=0, zorder=0,)

        # Vertical arrow + label
        x_arrow = xarrow[f_ref]
        y_arrow_top = f_ref*1.03
        y_arrow_bottom = f_ref*0.73

        ax.annotate( "", xy=(x_arrow, y_arrow_bottom), xytext=(x_arrow, y_arrow_top), arrowprops=dict(arrowstyle="->", color="0.35", lw=1.), annotation_clip=False, )  

        ax.text( x_arrow + 0.05, np.sqrt(y_arrow_top * y_arrow_bottom),  arrow_label[f_ref], color="0.35", fontsize=9, rotation=0, va="center", ha="left", zorder=10, )


    ax.set_xticks(x_positions)
    ax.set_xticklabels([rf"$\alpha={alpha}$" for alpha in alphas])

    ax.set_yscale("log")
    #ax.set_ylabel(
    #    r"$\int R_{\rm AGN}(z)\,dz \,/\, "
    #    r"\int R_{\rm LVK,DefaultBBH}(z)\,dz$"
    #)
    ax.set_ylabel(r"$f_{\rm AGN}$ from MRD integral")
    ax.set_xlabel(r"AGN disk viscosity parameter")
    ax.set_xlim(-0.5, 1.5)

    # Robust y-limits
    valid_arrays = [
        fractions_by_alpha[alpha]
        for alpha in alphas
        if len(fractions_by_alpha[alpha]) > 0
    ]

    if len(valid_arrays) > 0:
        all_f = np.concatenate(valid_arrays)
        all_f = all_f[np.isfinite(all_f) & (all_f > 0)]

        if len(all_f) > 0:
            ax.set_ylim(
                0.5 * np.nanmin(all_f),
                2.0 * max(1.0, np.nanmax(all_f)),
            )


    # Custom legend explaining the intervals
    legend_handles = [
        Line2D([0], [0], color="black", linestyle=":", linewidth=0.8, alpha=0.8, label=r"$\mathcal{R}_{\rm AGN}=\mathcal{R}_{\rm GWTC5}$ (DefaultBBH)",),
        Patch(facecolor="gray", alpha=0.6, edgecolor='0.35', label=r"Constraints from spatial correlations"),  
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    fig.tight_layout()

    outfile = f"plots/MRD_integrated_fraction_by_alpha_{base_model}.png"
    fig.savefig(outfile, bbox_inches="tight", dpi=300)

    print(f"Saved {outfile}")
    plt.show()
