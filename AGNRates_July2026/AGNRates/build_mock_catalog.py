import numpy as np
import pandas as pd

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

def observer_rate_density_dNdz(
    z_grid,
    Rz,
    duration_yr=1.0,
    epsilon_z=None,
    sky_fraction=1.0,
    cosmology=cosmo,
):
    """
    Compute dN_obs/dz for an observing time duration_yr.

    Parameters
    ----------
    z_grid : array-like
        Redshift grid.

    Rz : array-like
        Source-frame merger rate density R(z)
        in Gpc^-3 yr^-1.

    duration_yr : float
        Observer-frame observing time in yr.

    epsilon_z : array-like or None
        Detection efficiency as a function of z.
        If None, assumes epsilon(z)=1.

    sky_fraction : float
        Fraction of sky observed. For ET, usually use 1.

    cosmology : astropy cosmology
        Cosmology used to compute dVc/dz.

    Returns
    -------
    dNdz : ndarray
        Observer-frame number of events per unit redshift.
    """

    z_grid = np.asarray(z_grid, dtype=float)
    Rz = np.asarray(Rz, dtype=float)

    if epsilon_z is None:
        epsilon_z = np.ones_like(z_grid)
    else:
        epsilon_z = np.asarray(epsilon_z, dtype=float)

    if len(z_grid) != len(Rz):
        raise ValueError("z_grid and Rz must have the same length.")

    if len(z_grid) != len(epsilon_z):
        raise ValueError("z_grid and epsilon_z must have the same length.")

    # dVc/dz per steradian in Mpc^3 / sr
    dVc_dz_dOmega = cosmology.differential_comoving_volume(z_grid)

    # all-sky dVc/dz in Gpc^3
    dVc_dz = (
        dVc_dz_dOmega
        * 4.0
        * np.pi
        * u.sr
    ).to(u.Gpc**3).value

    dVc_dz *= sky_fraction

    dNdz = duration_yr * Rz / (1.0 + z_grid) * dVc_dz * epsilon_z

    dNdz = np.nan_to_num(dNdz, nan=0.0, posinf=0.0, neginf=0.0)
    dNdz[dNdz < 0] = 0.0

    return dNdz


def make_mock_catalog_from_Rz(
    df,
    z_grid,
    Rz,
    z_column="z_merg",
    weights=None,
    duration_yr=1.0,
    epsilon_z=None,
    sky_fraction=1.0,
    z_min=None,
    z_max=None,
    random_state=None,
    replace=True,
    label=None,
):
    """
    Draw a finite observer-frame mock catalog using a redshift-dependent
    merger rate density R(z).

    The expected number of detected/observable mergers is

        lambda = integral dz [
            T_obs * R(z)/(1+z) * dVc/dz * epsilon(z)
        ]

    and the catalog size is

        N_mock ~ Poisson(lambda).

    Parameters
    ----------
    df : pandas.DataFrame
        Intrinsic population/library of simulated mergers.
        Must contain a redshift column, e.g. 'z_merg'.

    z_grid : array-like
        Redshift grid where R(z) is evaluated.

    Rz : array-like
        Source-frame merger rate density in Gpc^-3 yr^-1.

    z_column : str
        Name of the redshift column in df.

    weights : array-like or None
        Optional row weights for df. These can encode the mixture over
        SMBH mass, f_Edd, alpha, etc.

    duration_yr : float
        Observer-frame observing time in yr.

    epsilon_z : array-like or None
        Detection efficiency on z_grid. If None, epsilon(z)=1.

    sky_fraction : float
        Fraction of sky observed. Usually 1 for ET.

    z_min, z_max : float or None
        Optional redshift cuts.

    random_state : int, np.random.Generator, or None
        Random seed or generator.

    replace : bool
        Sample with replacement. Usually True for Monte Carlo libraries.

    label : str or None
        Optional label added as 'mock_label'.

    Returns
    -------
    mock_df : pandas.DataFrame
        Finite mock catalog.

    info : dict
        Metadata about the draw.

    rate_df : pandas.DataFrame
        Redshift-dependent observer-frame rate information.
    """

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    df = df.copy()

    if z_column not in df.columns:
        raise ValueError(f"df does not contain z_column='{z_column}'.")

    z_grid = np.asarray(z_grid, dtype=float)
    Rz = np.asarray(Rz, dtype=float)

    if epsilon_z is None:
        epsilon_z = np.ones_like(z_grid)
    else:
        epsilon_z = np.asarray(epsilon_z, dtype=float)

    # Optional redshift cuts on the rate integral
    rate_mask = np.ones_like(z_grid, dtype=bool)

    if z_min is not None:
        rate_mask &= z_grid >= z_min

    if z_max is not None:
        rate_mask &= z_grid <= z_max

    z_use = z_grid[rate_mask]
    Rz_use = Rz[rate_mask]
    eps_use = epsilon_z[rate_mask]

    if len(z_use) < 2:
        raise ValueError("Need at least two redshift points after applying cuts.")

    dNdz = observer_rate_density_dNdz(
        z_grid=z_use,
        Rz=Rz_use,
        duration_yr=duration_yr,
        epsilon_z=eps_use,
        sky_fraction=sky_fraction,
    )

    lambda_expected = np.trapz(dNdz, z_use)

    if lambda_expected < 0 or not np.isfinite(lambda_expected):
        raise ValueError(f"Invalid lambda_expected={lambda_expected}.")

    n_mock = rng.poisson(lambda_expected)

    rate_df = pd.DataFrame(
        {
            "z": z_use,
            "Rz_Gpc3_yr": Rz_use,
            "epsilon_z": eps_use,
            "dNobs_dz": dNdz,
        }
    )

    if n_mock == 0:
        mock_df = df.iloc[[]].copy()
        mock_df["mock_event_id"] = []
        if label is not None:
            mock_df["mock_label"] = label

        info = {
            "duration_yr": duration_yr,
            "sky_fraction": sky_fraction,
            "z_min": z_min,
            "z_max": z_max,
            "lambda_expected": lambda_expected,
            "N_mock": n_mock,
        }

        print("lambda_expected =", info["lambda_expected"])
        print("N_mock drawn =", info["N_mock"])
        
        print("Rz min/max =", np.nanmin(Rz), np.nanmax(Rz))
        print("z min/max used =", rate_df["z"].min(), rate_df["z"].max())
        print("Integral of dNobs/dz =", np.trapz(rate_df["dNobs_dz"], rate_df["z"]))
        
        print(rate_df[["z", "Rz_Gpc3_yr", "epsilon_z", "dNobs_dz"]].head())
        print(rate_df[["z", "Rz_Gpc3_yr", "epsilon_z", "dNobs_dz"]].tail())
    
        return mock_df, info, rate_df

    # ------------------------------------------------------------------
    # Step 1: sample mock event redshifts from dN/dz.
    #
    # Use bin-integrated weights, not just dNdz values, so the sampling is
    # consistent with lambda = integral dNdz dz.
    # ------------------------------------------------------------------

    z_edges = np.zeros(len(z_use) + 1)
    z_edges[1:-1] = 0.5 * (z_use[1:] + z_use[:-1])
    z_edges[0] = z_use[0] - 0.5 * (z_use[1] - z_use[0])
    z_edges[-1] = z_use[-1] + 0.5 * (z_use[-1] - z_use[-2])
    z_edges[0] = max(0.0, z_edges[0])

    dz = np.diff(z_edges)

    dz_weights = dNdz * dz
    dz_weights = np.nan_to_num(dz_weights, nan=0.0, posinf=0.0, neginf=0.0)
    dz_weights[dz_weights < 0] = 0.0

    dz_sum = dz_weights.sum()

    if dz_sum <= 0:
        raise ValueError("dNdz integrates to zero. No events can be sampled.")

    pz = dz_weights / dz_sum

    sampled_bin_indices = rng.choice(
        np.arange(len(z_use)),
        size=n_mock,
        replace=True,
        p=pz,
    )

    # Draw continuously within each selected redshift bin,
    # instead of placing every event exactly at the grid center.
    sampled_z = rng.uniform(
        z_edges[sampled_bin_indices],
        z_edges[sampled_bin_indices + 1],
    )

    # ------------------------------------------------------------------
    # Step 2: sample population-library events close to those redshifts.
    #
    # This assumes df already contains events assigned to z_merg.
    # We bin df in redshift and sample from the corresponding bin.
    # ------------------------------------------------------------------

    z_edges = np.zeros(len(z_use) + 1)
    z_edges[1:-1] = 0.5 * (z_use[1:] + z_use[:-1])
    z_edges[0] = z_use[0] - 0.5 * (z_use[1] - z_use[0])
    z_edges[-1] = z_use[-1] + 0.5 * (z_use[-1] - z_use[-2])

    z_edges[0] = max(0.0, z_edges[0])

    df_z = df[z_column].to_numpy(dtype=float)

    if weights is None:
        base_weights = np.ones(len(df), dtype=float)
    else:
        base_weights = np.asarray(weights, dtype=float)

        if len(base_weights) != len(df):
            raise ValueError(
                f"weights has length {len(base_weights)}, "
                f"but df has length {len(df)}."
            )

        base_weights = np.nan_to_num(base_weights, nan=0.0, posinf=0.0, neginf=0.0)

        if np.any(base_weights < 0):
            raise ValueError("weights must be non-negative.")

    selected_indices = []
    selected_indices = []

    for z_event in sampled_z:
        iz = np.searchsorted(z_edges, z_event, side="right") - 1
        iz = np.clip(iz, 0, len(z_use) - 1)

        zlo = z_edges[iz]
        zhi = z_edges[iz + 1]

        in_bin = (
            np.isfinite(df_z)
            & (df_z >= zlo)
            & (df_z < zhi)
        )

        candidate_idx = np.where(in_bin)[0]

        if len(candidate_idx) == 0:
            finite = np.isfinite(df_z)
            if not np.any(finite):
                raise ValueError(f"No finite values found in df['{z_column}'].")
            nearest = np.nanargmin(np.abs(df_z - z_event))
            candidate_idx = np.array([nearest])

        w = base_weights[candidate_idx]
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        if w.sum() <= 0:
            p = None
        else:
            p = w / w.sum()

        chosen = rng.choice(candidate_idx, size=1, replace=replace, p=p)[0]
        selected_indices.append(chosen)
    
    mock_df = df.iloc[selected_indices].copy()
    mock_df = mock_df.reset_index(drop=True)

    mock_df["mock_event_id"] = np.arange(len(mock_df))
    mock_df["z_mock_rate_draw"] = sampled_z

    if label is not None:
        mock_df["mock_label"] = label

    info = {
        "duration_yr": duration_yr,
        "sky_fraction": sky_fraction,
        "z_min": z_min,
        "z_max": z_max,
        "lambda_expected": lambda_expected,
        "N_mock": n_mock,
    }

    return mock_df, info, rate_df


############################## Main ##############################

from pathlib import Path
import json
import argparse


def read_table(path):
    """
    Read a table from csv, txt/ascii, parquet, hdf/h5, or feather.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".txt", ".dat", ".asc"]:
        # First try whitespace-separated columns.
        # This works for files like:
        # z_merg M1 M2 chi_eff ...
        try:
            return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        except Exception:
            # Fallback: comma-separated text file.
            return pd.read_csv(path, comment="#")

    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    if suffix in [".h5", ".hdf", ".hdf5"]:
        return pd.read_hdf(path)

    if suffix == ".feather":
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def read_mrd_file(
    mrd_path,
    z_col="z",
    Rz_col="Rz",
):
    """
    Read merger-rate-density curve.

    Expected columns:
        z_col  : redshift
        Rz_col : source-frame merger rate density in Gpc^-3 yr^-1
    """
    mrd = read_table(mrd_path)

    if z_col not in mrd.columns:
        raise ValueError(
            f"MRD file does not contain z_col='{z_col}'. "
            f"Available columns: {list(mrd.columns)}"
        )

    if Rz_col not in mrd.columns:
        raise ValueError(
            f"MRD file does not contain Rz_col='{Rz_col}'. "
            f"Available columns: {list(mrd.columns)}"
        )

    z_grid = mrd[z_col].to_numpy(dtype=float)
    Rz = mrd[Rz_col].to_numpy(dtype=float)

    good = np.isfinite(z_grid) & np.isfinite(Rz) & (Rz >= 0)

    z_grid = z_grid[good]
    Rz = Rz[good]

    order = np.argsort(z_grid)
    z_grid = z_grid[order]
    Rz = Rz[order]

    return z_grid, Rz, mrd


def read_epsilon_file(
    epsilon_path,
    z_grid,
    z_col="z",
    eps_col="epsilon_z",
):
    """
    Read detection efficiency epsilon(z) and interpolate it onto z_grid.

    Expected columns:
        z_col   : redshift
        eps_col : detection efficiency in [0, 1]
    """
    if epsilon_path is None:
        return None

    eps_df = read_table(epsilon_path)

    if z_col not in eps_df.columns:
        raise ValueError(
            f"epsilon file does not contain z_col='{z_col}'. "
            f"Available columns: {list(eps_df.columns)}"
        )

    if eps_col not in eps_df.columns:
        raise ValueError(
            f"epsilon file does not contain eps_col='{eps_col}'. "
            f"Available columns: {list(eps_df.columns)}"
        )

    z_eps = eps_df[z_col].to_numpy(dtype=float)
    eps = eps_df[eps_col].to_numpy(dtype=float)

    good = np.isfinite(z_eps) & np.isfinite(eps)

    z_eps = z_eps[good]
    eps = eps[good]

    order = np.argsort(z_eps)
    z_eps = z_eps[order]
    eps = eps[order]

    eps = np.clip(eps, 0.0, 1.0)

    epsilon_z = np.interp(
        z_grid,
        z_eps,
        eps,
        left=0.0,
        right=0.0,
    )

    return epsilon_z


def get_event_weights(df, weight_col=None):
    """
    Get row weights for the event catalog.

    If weight_col is None, sample uniformly within each redshift bin.
    """
    if weight_col is None:
        return None

    if weight_col not in df.columns:
        raise ValueError(
            f"Catalog does not contain weight_col='{weight_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    weights = df[weight_col].to_numpy(dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(weights < 0):
        raise ValueError(f"Column '{weight_col}' contains negative weights.")

    if weights.sum() <= 0:
        raise ValueError(f"Column '{weight_col}' sums to zero.")

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Generate a finite ET-like mock catalog from R(z)."
    )

    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to event population catalog. Must contain z_column.",
    )

    parser.add_argument(
        "--mrd",
        required=True,
        help="Path to merger-rate-density table.",
    )

    parser.add_argument(
        "--outdir",
        default="../outputs/mock_catalog_output",
        help="Output directory.",
    )

    parser.add_argument(
        "--z-column",
        default="z_merg",
        help="Redshift column in the event catalog.",
    )

    parser.add_argument(
        "--mrd-z-column",
        default="z",
        help="Redshift column in the MRD file.",
    )

    parser.add_argument(
        "--mrd-rate-column",
        default="Rz",
        help="R(z) column in the MRD file, in Gpc^-3 yr^-1.",
    )

    parser.add_argument(
        "--weight-column",
        default=None,
        help="Optional row-weight column in the event catalog.",
    )

    parser.add_argument(
        "--epsilon-file",
        default=None,
        help="Optional table containing ET detection efficiency epsilon(z).",
    )

    parser.add_argument(
        "--epsilon-z-column",
        default="z",
        help="Redshift column in epsilon file.",
    )

    parser.add_argument(
        "--epsilon-column",
        default="epsilon_z",
        help="epsilon(z) column in epsilon file.",
    )

    parser.add_argument(
        "--duration-yr",
        type=float,
        default=1.0,
        help="Observer-frame observing duration in yr.",
    )

    parser.add_argument(
        "--sky-fraction",
        type=float,
        default=1.0,
        help="Observed sky fraction. Use 1 for all sky.",
    )

    parser.add_argument(
        "--z-min",
        type=float,
        default=0.0,
        help="Minimum redshift.",
    )

    parser.add_argument(
        "--z-max",
        type=float,
        default=20.0,
        help="Maximum redshift.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--label",
        default="ET_1yr_allsky",
        help="Label stored in mock_label column.",
    )

    parser.add_argument(
        "--prefix",
        default="mock_catalog_ET_1yr_Rz",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Read event catalog
    # ------------------------------------------------------------
    print(f"Reading event catalog: {args.catalog}")
    df_population = read_table(args.catalog)

    if args.z_column not in df_population.columns:
        raise ValueError(
            f"Catalog does not contain z_column='{args.z_column}'. "
            f"Available columns: {list(df_population.columns)}"
        )

    print(f"Catalog contains {len(df_population):,} rows.")

    # ------------------------------------------------------------
    # 2. Read MRD curve R(z)
    # ------------------------------------------------------------
    print(f"Reading MRD curve: {args.mrd}")
    z_grid, Rz, mrd_df = read_mrd_file(
        args.mrd,
        z_col=args.mrd_z_column,
        Rz_col=args.mrd_rate_column,
    )

    print(
        f"MRD grid: {len(z_grid)} redshift points, "
        f"z=[{z_grid.min():.3g}, {z_grid.max():.3g}]"
    )

    # ------------------------------------------------------------
    # 3. Optional event weights
    # ------------------------------------------------------------
    event_weights = get_event_weights(
        df_population,
        weight_col=args.weight_column,
    )

    if event_weights is None:
        print("No event weights provided. Sampling uniformly within redshift bins.")
    else:
        print(f"Using event weights from column: {args.weight_column}")

    # ------------------------------------------------------------
    # 4. Optional ET efficiency epsilon(z)
    # ------------------------------------------------------------
    epsilon_z = read_epsilon_file(
        args.epsilon_file,
        z_grid=z_grid,
        z_col=args.epsilon_z_column,
        eps_col=args.epsilon_column,
    )

    if epsilon_z is None:
        print("No epsilon(z) file provided. Using epsilon(z)=1.")
    else:
        print(f"Using epsilon(z) from file: {args.epsilon_file}")

    # ------------------------------------------------------------
    # 5. Draw mock catalog
    # ------------------------------------------------------------
    mock_df, info, rate_df = make_mock_catalog_from_Rz(
        df=df_population,
        z_grid=z_grid,
        Rz=Rz,
        z_column=args.z_column,
        weights=event_weights,
        duration_yr=args.duration_yr,
        epsilon_z=epsilon_z,
        sky_fraction=args.sky_fraction,
        z_min=args.z_min,
        z_max=args.z_max,
        random_state=args.seed,
        label=args.label,
    )

    print("Mock catalog info:")
    for key, val in info.items():
        print(f"  {key}: {val}")

    # ------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------
    mock_path = outdir / f"{args.prefix}.csv"
    rate_path = outdir / f"{args.prefix}_rate_curve.csv"
    info_path = outdir / f"{args.prefix}_info.json"

    mock_df.to_csv(mock_path, index=False)
    rate_df.to_csv(rate_path, index=False)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Saved mock catalog to: {mock_path}")
    print(f"Saved rate curve to:    {rate_path}")
    print(f"Saved info to:          {info_path}")


if __name__ == "__main__":
    main()