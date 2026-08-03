import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ============================================================
# Helpers
# ============================================================

def find_col(df, candidates, required=True):
    """
    Return the first column in candidates that exists in df.
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            "Could not find any of these columns:\n"
            f"{candidates}\n\n"
            "Available columns are:\n"
            f"{list(df.columns)}"
        )

    return None


def compute_chirp_mass(m1, m2):
    """
    Source-frame chirp mass:
        Mchirp = (m1 m2)^(3/5) / (m1 + m2)^(1/5)
    """
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def add_mass_column(df, mass_quantity="m1", mass_col=None):
    """
    Add or identify the mass column to plot.

    Default is M1.
    """
    df = df.copy()

    if mass_col is not None:
        if mass_col not in df.columns:
            raise KeyError(
                f"mass_col='{mass_col}' not found.\n"
                f"Available columns are:\n{list(df.columns)}"
            )
        return df, mass_col, mass_col

    m1_col = find_col(
        df,
        ["c1:M1/Msun", "m1", "M1", "mass_1", "primary_mass"],
        required=False,
    )

    m2_col = find_col(
        df,
        ["c2:M2/Msun", "m2", "M2", "mass_2", "secondary_mass"],
        required=False,
    )

    if mass_quantity == "m1":
        if m1_col is None:
            raise KeyError("Could not infer M1 column.")
        return df, m1_col, r"$M_1\,[M_\odot]$"

    if mass_quantity == "m2":
        if m2_col is None:
            raise KeyError("Could not infer M2 column.")
        return df, m2_col, r"$M_2\,[M_\odot]$"

    if mass_quantity == "chirp":
        chirp_col = find_col(
            df,
            [
                "Mchirp",
                "M_chirp",
                "chirp_mass",
                "Mc",
                "M_c",
                "source_frame_chirp_mass",
                "detector_frame_chirp_mass",
            ],
            required=False,
        )

        if chirp_col is not None:
            return df, chirp_col, r"$\mathcal{M}_{\rm chirp}\,[M_\odot]$"

        if m1_col is None or m2_col is None:
            raise KeyError("Need M1 and M2 columns to compute chirp mass.")

        df["Mchirp"] = compute_chirp_mass(
            df[m1_col].to_numpy(float),
            df[m2_col].to_numpy(float),
        )

        return df, "Mchirp", r"$\mathcal{M}_{\rm chirp}\,[M_\odot]$"

    raise ValueError("mass_quantity must be one of: 'm1', 'm2', 'chirp'.")


def get_event_weights(
    df,
    weight_col=None,
    info_json=None,
    duration_yr=1.0,
):
    """
    Return event weights.

    Default:
        each row has weight 1.

    Options:
        weight_col:
            use an existing column as event weight.

        info_json:
            read lambda_expected from mock info file and assign each event
            weight = lambda_expected / N_mock.

    Notes:
        If duration_yr=1, weighted counts are yr^-1.
        If duration_yr != 1, the plotted rate is weight / duration_yr.
    """

    n = len(df)

    if n == 0:
        raise ValueError("Input catalog has zero rows.")

    if weight_col is not None and weight_col.lower() != "none":
        if weight_col not in df.columns:
            raise KeyError(
                f"weight_col='{weight_col}' not found.\n"
                f"Available columns are:\n{list(df.columns)}"
            )

        weights = df[weight_col].to_numpy(float)
        label = rf"Weighted mergers per bin"

    elif info_json is not None:
        info_path = Path(info_json)

        if not info_path.exists():
            raise FileNotFoundError(f"Info JSON not found: {info_path}")

        with open(info_path, "r") as f:
            info = json.load(f)

        if "lambda_expected" not in info:
            raise KeyError(f"'lambda_expected' not found in {info_path}")

        lambda_expected = float(info["lambda_expected"])
        weights = np.full(n, lambda_expected / n)
        label = rf"Expected mergers per bin"

    else:
        weights = np.ones(n, dtype=float)
        label = rf"Mock mergers per bin"

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(weights < 0):
        raise ValueError("Weights contain negative values.")

    return weights, label


def finite_positive_range(x, lower_pct=0.3, upper_pct=99.7):
    """
    Return robust positive plotting range.
    """
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x) & (x > 0)

    if not np.any(good):
        raise ValueError("No finite positive values found.")

    xx = x[good]

    xmin = np.nanpercentile(xx, lower_pct)
    xmax = np.nanpercentile(xx, upper_pct)

    xmin = max(xmin, 1e-6)
    xmax = max(xmax, xmin * 1.01)

    return xmin, xmax


# ============================================================
# Plotting
# ============================================================

def plot_mock_catalog_counts(
    df,
    z_col,
    mass_col,
    mass_label,
    weights,
    count_label,
    duration_yr=1.0,
    z_bins=None,
    mass_bins=None,
    max_scatter_points=30000,
    log_2d=True,
    log_mass_axis=True,
    title=None,
    output=None,
):
    """
    Plot N(z), N(M), scatter, and 2D N(z, M).

    This plots COUNTS PER BIN, not PDFs.

    If weights are all 1:
        H gives number of mock catalog events per bin.

    If weights sum to lambda_expected:
        H gives expected number of mergers per bin over duration_yr.

    If duration_yr != 1:
        H_rate = H / duration_yr gives yr^-1 per bin.
    """

    z = df[z_col].to_numpy(float)
    mass = df[mass_col].to_numpy(float)
    weights = np.asarray(weights, dtype=float)

    good = (
        np.isfinite(z)
        & np.isfinite(mass)
        & np.isfinite(weights)
        & (z >= 0)
        & (mass > 0)
        & (weights >= 0)
    )

    z = z[good]
    mass = mass[good]
    weights = weights[good]

    if len(z) == 0:
        raise ValueError("No valid events after filtering.")

    # Convert from counts over duration_yr to rate per yr if needed.
    rate_weights = weights / duration_yr

    # ------------------------------------------------------------
    # Histograms: COUNTS / RATES, not PDF
    # ------------------------------------------------------------
    Nz, z_edges = np.histogram(
        z,
        bins=z_bins,
        weights=rate_weights,
        density=False,
    )

    Nm, m_edges = np.histogram(
        mass,
        bins=mass_bins,
        weights=rate_weights,
        density=False,
    )

    H2, z_edges_2d, m_edges_2d = np.histogram2d(
        z,
        mass,
        bins=[z_bins, mass_bins],
        weights=rate_weights,
        density=False,
    )

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    if log_mass_axis:
        m_centers = np.sqrt(m_edges[:-1] * m_edges[1:])
    else:
        m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])

    # ------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------
    total_rate = np.nansum(rate_weights)
    total_hist_rate = np.nansum(H2)

    print("")
    print("Histogram sanity check")
    print("----------------------")
    print(f"Rows used:                 {len(z):,}")
    print(f"Sum of event rates:         {total_rate:.6g} yr^-1")
    print(f"Sum of 2D histogram rates:  {total_hist_rate:.6g} yr^-1")
    print(f"Lost outside bins:          {total_rate - total_hist_rate:.6g} yr^-1")
    print(f"Max 2D bin value:           {np.nanmax(H2):.6g} yr^-1 bin^-1")
    print("")

    # ------------------------------------------------------------
    # Scatter downsampling for display only
    # ------------------------------------------------------------
    rng = np.random.default_rng(42)

    if len(z) > max_scatter_points:
        idx = rng.choice(len(z), size=max_scatter_points, replace=False)
    else:
        idx = np.arange(len(z))

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        constrained_layout=True,
    )

    ax_z = axs[0, 0]
    ax_m = axs[0, 1]
    ax_scatter = axs[1, 0]
    ax_2d = axs[1, 1]

    # N(z)
    ax_z.step(
        z_centers,
        Nz,
        where="mid",
        linewidth=2,
    )
    ax_z.set_xlabel(r"$z_{\rm merg}$")
    ax_z.set_ylabel(r"$N_{\rm mergers}\ {\rm yr}^{-1}\ {\rm bin}^{-1}$")
    ax_z.set_title(r"$N(z_{\rm merg})$")
    ax_z.grid(alpha=0.3)

    # N(M)
    ax_m.step(
        m_centers,
        Nm,
        where="mid",
        linewidth=2,
    )
    if log_mass_axis:
        ax_m.set_xscale("log")
    ax_m.set_xlabel(mass_label)
    ax_m.set_ylabel(r"$N_{\rm mergers}\ {\rm yr}^{-1}\ {\rm bin}^{-1}$")
    ax_m.set_title(r"$N(M)$")
    ax_m.grid(alpha=0.3)

    # Scatter
    ax_scatter.scatter(
        z[idx],
        mass[idx],
        s=6,
        alpha=0.25,
        rasterized=True,
    )
    if log_mass_axis:
        ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(r"$z_{\rm merg}$")
    ax_scatter.set_ylabel(mass_label)
    ax_scatter.set_title("Mock catalog events")
    ax_scatter.grid(alpha=0.3)

    # 2D N(z, M)
    H_plot = H2.T

    if log_2d:
        positive = H_plot[H_plot > 0]
        if len(positive) > 0:
            norm = LogNorm(
                vmin=np.nanmin(positive),
                vmax=np.nanmax(positive),
            )
        else:
            norm = None
    else:
        norm = None

    pcm = ax_2d.pcolormesh(
        z_edges_2d,
        m_edges_2d,
        H_plot,
        shading="auto",
        norm=norm,
    )

    if log_mass_axis:
        ax_2d.set_yscale("log")

    ax_2d.set_xlabel(r"$z_{\rm merg}$")
    ax_2d.set_ylabel(mass_label)
    ax_2d.set_title(r"$N(z_{\rm merg}, M)$")

    cbar = fig.colorbar(pcm, ax=ax_2d)
    cbar.set_label(r"$N_{\rm mergers}\ {\rm yr}^{-1}\ {\rm bin}^{-1}$")

    if title is None:
        title = rf"Mock catalog counts, total $\simeq {total_hist_rate:.3g}\ {{\rm yr}}^{{-1}}$"

    fig.suptitle(title, fontsize=15)

    if output is not None:
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"Saved figure to: {output}")

    return fig, axs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot mock BBH catalog counts in redshift and mass bins. "
            "This script plots N, not a normalized PDF."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to mock catalog CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="mock_catalog_output/figures",
        help="Directory where the figure is saved.",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output filename. If omitted, one is generated.",
    )

    parser.add_argument(
        "--z-col",
        default=None,
        help=(
            "Redshift column. If omitted, inferred. "
            "For mock catalogs, z_mock_rate_draw is preferred when available."
        ),
    )

    parser.add_argument(
        "--mass-quantity",
        default="m1",
        choices=["m1", "m2", "chirp"],
        help="Mass quantity to plot. Default: m1.",
    )

    parser.add_argument(
        "--mass-col",
        default=None,
        help="Explicit mass column to plot. Overrides --mass-quantity.",
    )

    parser.add_argument(
        "--weight-col",
        default="none",
        help=(
            "Optional weight column. Use 'none' for raw mock event counts. "
            "Default: none."
        ),
    )

    parser.add_argument(
        "--info-json",
        default=None,
        help=(
            "Optional mock info JSON containing lambda_expected. "
            "If provided and --weight-col none, each event is assigned "
            "lambda_expected / N_mock."
        ),
    )

    parser.add_argument(
        "--duration-yr",
        type=float,
        default=1.0,
        help=(
            "Catalog duration in observer-frame years. "
            "The plotted quantity is divided by this to give yr^-1."
        ),
    )

    parser.add_argument(
        "--zmin",
        type=float,
        default=0.0,
        help="Minimum redshift shown.",
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="Maximum redshift shown. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--mmin",
        type=float,
        default=None,
        help="Minimum mass shown. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--mmax",
        type=float,
        default=None,
        help="Maximum mass shown. If omitted, inferred from data.",
    )

    parser.add_argument(
        "--nz",
        type=int,
        default=40,
        help="Number of redshift bins.",
    )

    parser.add_argument(
        "--nm",
        type=int,
        default=40,
        help="Number of mass bins.",
    )

    parser.add_argument(
        "--linear-mass-bins",
        action="store_true",
        help="Use linear mass bins instead of log-spaced bins.",
    )

    parser.add_argument(
        "--linear-mass-axis",
        action="store_true",
        help="Use a linear mass axis instead of log mass axis.",
    )

    parser.add_argument(
        "--no-log-2d",
        action="store_true",
        help="Disable logarithmic color scale for the 2D count map.",
    )

    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=30000,
        help="Maximum number of scatter points shown. Histograms use all rows.",
    )

    args = parser.parse_args()

    if args.duration_yr <= 0:
        raise ValueError("--duration-yr must be positive.")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input catalog not found: {input_path}")

    print(f"Reading catalog: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------
    # Redshift column
    # ------------------------------------------------------------
    if args.z_col is None:
        z_col = find_col(
            df,
            [
                "z_mock_rate_draw",
                "z_merg",
                "zmerg",
                "z_merge",
                "merger_redshift",
                "redshift",
                "z",
            ],
        )
    else:
        z_col = args.z_col
        if z_col not in df.columns:
            raise KeyError(
                f"z_col='{z_col}' not found.\n"
                f"Available columns are:\n{list(df.columns)}"
            )

    # ------------------------------------------------------------
    # Mass column
    # ------------------------------------------------------------
    df, mass_col, mass_label = add_mass_column(
        df,
        mass_quantity=args.mass_quantity,
        mass_col=args.mass_col,
    )

    # ------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------
    weight_col = None if args.weight_col.lower() == "none" else args.weight_col

    weights, count_label = get_event_weights(
        df,
        weight_col=weight_col,
        info_json=args.info_json,
        duration_yr=args.duration_yr,
    )

    # ------------------------------------------------------------
    # Clean and apply plotting cuts
    # ------------------------------------------------------------
    z = df[z_col].to_numpy(float)
    mass = df[mass_col].to_numpy(float)

    good = (
        np.isfinite(z)
        & np.isfinite(mass)
        & np.isfinite(weights)
        & (z >= args.zmin)
        & (mass > 0)
        & (weights >= 0)
    )

    if args.zmax is not None:
        good &= z <= args.zmax

    if args.mmin is not None:
        good &= mass >= args.mmin

    if args.mmax is not None:
        good &= mass <= args.mmax

    df_plot = df.loc[good].copy()
    weights_plot = weights[good]

    if len(df_plot) == 0:
        raise ValueError("No events left after plotting cuts.")

    z_plot = df_plot[z_col].to_numpy(float)
    m_plot = df_plot[mass_col].to_numpy(float)

    print("")
    print("Plotting choices")
    print("----------------")
    print(f"Redshift column:     {z_col}")
    print(f"Mass column:         {mass_col}")
    print(f"Mass quantity:       {args.mass_quantity}")
    print(f"Weight column:       {weight_col}")
    print(f"Info JSON:           {args.info_json}")
    print(f"Duration:            {args.duration_yr} yr")
    print(f"Rows after cuts:     {len(df_plot):,}")
    print(f"z range:             {np.nanmin(z_plot):.4g} -- {np.nanmax(z_plot):.4g}")
    print(f"mass range:          {np.nanmin(m_plot):.4g} -- {np.nanmax(m_plot):.4g}")
    print(f"Sum weights:         {np.nansum(weights_plot):.6g}")
    print(f"Sum weights / yr:    {np.nansum(weights_plot) / args.duration_yr:.6g} yr^-1")

    # ------------------------------------------------------------
    # Bins
    # ------------------------------------------------------------
    zmin = args.zmin

    if args.zmax is not None:
        zmax = args.zmax
    else:
        zmax = np.nanpercentile(z_plot, 99.7)
        zmax = max(zmax, np.nanmax(z_plot))
        zmax = max(zmax, zmin + 1e-3)

    z_bins = np.linspace(zmin, zmax, args.nz + 1)

    if args.mmin is not None:
        mmin = args.mmin
    else:
        mmin, _ = finite_positive_range(m_plot)

    if args.mmax is not None:
        mmax = args.mmax
    else:
        _, mmax = finite_positive_range(m_plot)

    mmin = max(mmin, 1e-6)
    mmax = max(mmax, mmin * 1.01)

    if args.linear_mass_bins:
        mass_bins = np.linspace(mmin, mmax, args.nm + 1)
    else:
        mass_bins = np.logspace(np.log10(mmin), np.log10(mmax), args.nm + 1)

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    mass_tag = args.mass_col if args.mass_col is not None else args.mass_quantity
    weight_tag = "raw_counts" if weight_col is None and args.info_json is None else (
        weight_col if weight_col is not None else "lambda_expected"
    )

    if args.output_name is None:
        output_name = f"mock_catalog_N_z_{mass_tag}_{weight_tag}.png"
    else:
        output_name = args.output_name

    output_path = output_dir / output_name

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    title = (
        rf"Mock catalog: $N(z,$ {mass_label}$)$ "
        rf"with total $\simeq {np.nansum(weights_plot) / args.duration_yr:.3g}\,{{\rm yr}}^{{-1}}$"
    )

    fig, axs = plot_mock_catalog_counts(
        df=df_plot,
        z_col=z_col,
        mass_col=mass_col,
        mass_label=mass_label,
        weights=weights_plot,
        count_label=count_label,
        duration_yr=args.duration_yr,
        z_bins=z_bins,
        mass_bins=mass_bins,
        max_scatter_points=args.max_scatter_points,
        log_2d=not args.no_log_2d,
        log_mass_axis=not args.linear_mass_axis,
        title=title,
        output=str(output_path),
    )

    plt.show()


if __name__ == "__main__":
    main()