"""
Plot p(M1) in different z_merg slices, joyplot style.

Example usage
-------------
python catalog_plotting.py \
    --input "/path/to/catalogs/*.csv" \
    --output pM1_joyplot_zmerg.png \
    --m1-col m1 \
    --z-col z_merg \
    --weight-col weight_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def weighted_quantile(x, q, weights=None):
    """
    Weighted quantile of x.

    Parameters
    ----------
    x : array
        Data values.
    q : float or array
        Quantile(s), between 0 and 1.
    weights : array or None
        Weights. If None, use unweighted quantiles.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)

    if weights is None:
        return np.quantile(x, q)

    weights = np.asarray(weights, dtype=float)

    good = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    x = x[good]
    weights = weights[good]

    if len(x) == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(q, cdf, x)


def weighted_kde_1d(x, weights, grid, bandwidth=None):
    """
    Simple weighted Gaussian KDE.

    This avoids requiring scipy. For very large catalogs, downsample first or
    replace this with scipy.stats.gaussian_kde.
    """
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    grid = np.asarray(grid, dtype=float)

    good = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    x = x[good]
    weights = weights[good]

    if len(x) < 2:
        return np.zeros_like(grid)

    weights = weights / np.sum(weights)

    if bandwidth is None:
        # Weighted Silverman-like bandwidth.
        q16, q84 = weighted_quantile(x, [0.16, 0.84], weights)
        sigma = 0.5 * (q84 - q16)

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.std(x)

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        neff = 1.0 / np.sum(weights**2)
        bandwidth = 1.06 * sigma * neff ** (-1.0 / 5.0)

    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    # KDE: sum_i w_i N(grid | x_i, h)
    u = (grid[:, None] - x[None, :]) / bandwidth
    dens = np.exp(-0.5 * u**2) @ weights
    dens /= np.sqrt(2.0 * np.pi) * bandwidth

    return dens


def auto_find_column(df, candidates, label):
    """
    Find the first available column among candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {label} column. Tried {candidates}. "
        f"Available columns are: {list(df.columns)}"
    )


def read_catalogs(input_pattern, sep=None):
    """
    Read one or many catalog files.
    """
    paths = sorted(glob.glob(str(input_pattern)))

    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched input pattern: {input_pattern}")

    dfs = []

    for path in paths:
        path = Path(path)

        if sep is None:
            # Let pandas infer separator.
            df = pd.read_csv(path, sep=None, engine="python")
        else:
            df = pd.read_csv(path, sep=sep)

        df["source_file"] = str(path)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def make_z_bins(df, z_col, z_edges=None, n_z_bins=8):
    """
    Define z_merg bins.
    """
    z = df[z_col].to_numpy(dtype=float)
    z = z[np.isfinite(z)]

    if z_edges is not None:
        edges = np.asarray(z_edges, dtype=float)
    else:
        zmin = np.nanmin(z)
        zmax = np.nanmax(z)

        if zmin == zmax:
            dz = 0.05 if zmin == 0 else 0.05 * abs(zmin)
            zmin -= dz
            zmax += dz

        edges = np.linspace(zmin, zmax, n_z_bins + 1)

    if len(edges) < 2:
        raise ValueError("Need at least two z-bin edges.")

    return edges


def plot_joyplot(
    df,
    m1_col,
    z_col,
    weight_col,
    output,
    z_edges=None,
    n_z_bins=8,
    m1_min=None,
    m1_max=None,
    log_m1=False,
    bandwidth=None,
    overlap=0.85,
    figsize=(7.0, 7.0),
    dpi=250,
):
    """
    Plot weighted p(M1) in z_merg slices.
    """
    data = df.copy()

    # Basic cleaning.
    data[m1_col] = pd.to_numeric(data[m1_col], errors="coerce")
    data[z_col] = pd.to_numeric(data[z_col], errors="coerce")
    data[weight_col] = pd.to_numeric(data[weight_col], errors="coerce")

    data = data[
        np.isfinite(data[m1_col])
        & np.isfinite(data[z_col])
        & np.isfinite(data[weight_col])
        & (data[weight_col] > 0)
        & (data[m1_col] > 0)
    ].copy()

    if len(data) == 0:
        raise ValueError("No valid rows after cleaning.")

    z_edges = make_z_bins(data, z_col, z_edges=z_edges, n_z_bins=n_z_bins)

    m1 = data[m1_col].to_numpy(dtype=float)

    if m1_min is None:
        m1_min = np.nanpercentile(m1, 0.5)
    if m1_max is None:
        m1_max = np.nanpercentile(m1, 99.5)

    if log_m1:
        x_data = np.log10(data[m1_col].to_numpy(dtype=float))
        x_min = np.log10(m1_min)
        x_max = np.log10(m1_max)
        x_grid = np.linspace(x_min, x_max, 600)
        x_plot = 10.0**x_grid
        xlabel = r"$m_1\,[M_\odot]$"
    else:
        x_data = data[m1_col].to_numpy(dtype=float)
        x_grid = np.linspace(m1_min, m1_max, 600)
        x_plot = x_grid
        xlabel = r"$m_1\,[M_\odot]$"

    data["_x_for_kde"] = x_data

    cmap = plt.get_cmap("coolwarm")

    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    norm = mpl.colors.Normalize(
        vmin=np.nanmin(z_centers),
        vmax=np.nanmax(z_centers),
    )

    fig, ax = plt.subplots(figsize=figsize)

    y_step = 1.0
    y_positions = []

    for i in range(len(z_edges) - 1):
        zlo = z_edges[i]
        zhi = z_edges[i + 1]

        in_bin = (data[z_col] >= zlo) & (data[z_col] < zhi)
        sub = data.loc[in_bin]

        if len(sub) < 2:
            continue

        x = sub["_x_for_kde"].to_numpy(dtype=float)
        w = sub[weight_col].to_numpy(dtype=float)

        density = weighted_kde_1d(
            x=x,
            weights=w,
            grid=x_grid,
            bandwidth=bandwidth,
        )

        if not np.any(np.isfinite(density)) or np.nanmax(density) <= 0:
            continue

        # Normalize each ridge to comparable height.
        density = density / np.nanmax(density)

        y0 = i * y_step * overlap
        y_positions.append((y0, zlo, zhi))

        z_mid = 0.5 * (zlo + zhi)
        color = cmap(norm(z_mid))


        floor_frac = 1e-3
        floor = floor_frac * np.nanmax(density)

        density_plot = np.log10(np.maximum(density, floor))
        density_plot -= np.log10(floor)

        if np.nanmax(density_plot) > 0:
            density_plot /= np.nanmax(density_plot)

        ridge_height = 0.85
        ax.fill_between(
            x_plot,
            y0,
            y0 + ridge_height * density_plot,
            color=color,
            alpha=0.65,
            linewidth=0,
        )

        ax.plot(
            x_plot,
            y0 + ridge_height * density_plot,
            color=color,
            linewidth=1.0,
        )

        # Weighted median marker.
        med = weighted_quantile(sub[m1_col].to_numpy(dtype=float), 0.5, w)

        if np.isfinite(med):
            ax.plot(
                [med, med],
                [y0, y0 + 0.95],
                color=color,
                linewidth=1.0,
                alpha=0.9,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"log PDF (normalized in each z slice)")

    if log_m1:
        ax.set_xscale("log")

    # Put redshift-bin labels on y-axis.
    yticks = []
    ylabels = []

    #for y0, zlo, zhi in y_positions:
    #    yticks.append(y0 + 0.15)
    #    ylabels.append(rf"${zlo:.2f} \leq z_{{\rm merg}} < {zhi:.2f}$")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_ylim(-0.1, max([y for y, _, _ in y_positions], default=0) + 1.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(r"Weighted $p(m_1)$ in $z_{\rm merg}$ slices")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$z_{\rm merg}$")

    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    print(f"Saved figure to {output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file or glob pattern, e.g. 'catalogs/*.csv'.",
    )
    parser.add_argument(
        "--output",
        default="../plots/pM1_joyplot_zmerg.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV separator. If omitted, pandas tries to infer it.",
    )
    parser.add_argument(
        "--m1-col",
        default=None,
        help="Primary-mass column. If omitted, try common names.",
    )
    parser.add_argument(
        "--z-col",
        default="z_merg",
        help="Merger-redshift column.",
    )
    parser.add_argument(
        "--weight-col",
        default=None,
        help="Weight column. If omitted, try weight_raw then weight_norm.",
    )
    parser.add_argument(
        "--n-z-bins",
        type=int,
        default=8,
        help="Number of z_merg slices if --z-edges is not provided.",
    )
    parser.add_argument(
        "--z-edges",
        type=float,
        nargs="+",
        default=None,
        help="Explicit z_merg bin edges, e.g. --z-edges 0 0.5 1 2 4 8.",
    )
    parser.add_argument(
        "--m1-min",
        type=float,
        default=None,
        help="Minimum M1 for plotting.",
    )
    parser.add_argument(
        "--m1-max",
        type=float,
        default=None,
        help="Maximum M1 for plotting.",
    )
    parser.add_argument(
        "--log-m1",
        action="store_true",
        help="Plot M1 on a log x-axis and compute KDE in log10(M1).",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="KDE bandwidth. In M1 units, or dex if --log-m1 is used.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.85,
        help="Vertical overlap between ridges. Smaller means more separation.",
    )

    args = parser.parse_args()

    df = read_catalogs(args.input, sep=args.sep)

    if args.m1_col is None:
        m1_col = auto_find_column(
            df,
            candidates=[
                "c1:M1/Msun",
                "m1",
                "M1",
            ],
            label="M1",
        )
    else:
        m1_col = args.m1_col

    if args.weight_col is None:
        weight_col = auto_find_column(
            df,
            candidates=[
                "weight_raw",
                "weight_norm",
                "weight",
                "weights",
            ],
            label="weight",
        )
    else:
        weight_col = args.weight_col

    plot_joyplot(
        df=df,
        m1_col=m1_col,
        z_col=args.z_col,
        weight_col=weight_col,
        output=args.output,
        z_edges=args.z_edges,
        n_z_bins=args.n_z_bins,
        m1_min=args.m1_min,
        m1_max=args.m1_max,
        log_m1=args.log_m1,
        bandwidth=args.bandwidth,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()