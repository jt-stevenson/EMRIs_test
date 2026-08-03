"""
The output catalogs are intrinsic event-property catalogs at fixed merger
redshift, i.e. they are meant to represent

    p(theta | z_merg in [z_low, z_high])

where theta are BBH properties.

The script uses weighted event copies. The same original BBH event may appear
in multiple redshift bins, with different z_form, z_merg, and weights.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import RegularGridInterpolator, interp1d


# =============================================================================
# GLOBAL USER SETTINGS
# =============================================================================

# Root directory/directories for the AGN grid.
# The script will construct run paths with the structure:
#   RUNS_ROOT / disk / alpha_{alpha} / logM_{Mbh} / fEdd_{fEdd} /
#       {torque}-3bb_{sigma}-{gh}-{progr}-tau_x_{lifetime} / outputs
RUNS_ROOTS = [ Path("/gpfs/bwfor/work/ws/hd_tn184-AGN_fastcluster_grid/RUNS_new_spin") ]

# AGN-grid dimensions used to construct the expected run directories.
GRID_DISKS = ["SG"]
GRID_ALPHAS_DEFAULT = ["0.1", "0.01"]
GRID_LOGM_BH = [f"{x:.1f}" for x in np.arange(5.0, 9.0, 0.1)]
GRID_FEDD = ["0.001", "0.01", "0.1", "1.", "10."]
GRID_PAIRUP = ["differential_migration"]
RUN_LABEL_DEFAULT = "G24_K18-3bb_0.0-IG25-agnostic-tau_x_1."

# Redshift model name for cosmological input files. 
REDSHIFT_MODEL_DEFAULT = "EL"  # e.g. "EL", "SE"
AGN_ABUNDANCE_DEFAULT = "LAM"  # e.g. "LAM", "HAM"

# Output directory.
OUTPUT_ROOT = Path("../outputs/catalogs_by_zmerg")

# Names of merger catalog files inside each run directory.
FIRST_GEN_FILENAME = "first_gen.txt"
NTH_GEN_FILENAME = "nth_generation.txt"

# Delay-time column in your merger catalogs.
# This must be the time from the beginning of the AGN episode to the BBH merger.
T_DELAY_COL = "c13:tmerg/Myr"
T_DELAY_UNIT = "Myr"

# Optional generation-number column if it exists.
# If it does not exist, the script will still add gen_flag based on the source file.
NGEN_COL = "c27:Ngen"

# Number of original Monte Carlo systems per AGN run.
# If this differs per run, either:
#   1. put N_sample in your run manifest, or
#   2. edit get_run_weight().
DEFAULT_N_SAMPLE = 1.0e4

# If each run should be weighted by an estimate of the number of BHs in the AGN,
# edit get_run_weight() below. Otherwise leave this as False.
INCLUDE_N_BH_FACTOR = False

# Maximum formation redshift to consider.
Z_FORM_MAX = 15.0

# Minimum formation redshift to consider.
# Usually 0 is fine. Formation redshift can be equal to merger redshift for very short delays.
Z_FORM_MIN = 0.0

# Number of formation-time quadrature points per event per redshift bin.
NZ_FORM_PER_EVENT = 1

# Redshift bins for the output merger catalogs.
Z_MERG_BINS = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
    (2.5, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
    (9.0, 10.0),
]

# Column separator for your input BBH merger catalogs.
INPUT_SEP = r"\s+"

# Column separator for output files.
OUTPUT_SEP = "\t"

# Choose whether p(fEdd|M,z) is tabulated in log10(fEdd) or fEdd.
PFEDD_USES_LOGFEDD = False

# If your single-AGN run directories encode parameters in their names,
# edit infer_run_parameters_from_name().
# Alternatively, create a manifest file and pass it with --manifest.
USE_DIRECTORY_NAME_TO_INFER_PARAMETERS = True


# =============================================================================
# Small utilities
# =============================================================================

def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def convert_delay_to_gyr(delay: np.ndarray | float, unit: str) -> np.ndarray | float:
    unit = unit.lower()
    if unit == "gyr":
        return delay
    if unit == "myr":
        return delay / 1.0e3
    if unit == "yr":
        return delay / 1.0e9
    raise ValueError(f"Unknown delay unit {unit!r}. Use 'yr', 'Myr', or 'Gyr'.")


def format_zbin_dir(z_low: float, z_high: float) -> str:
    return f"z_{z_low:.1f}_{z_high:.1f}"


def safe_float_from_regex(pattern: str, text: str, name: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(
            f"Could not infer {name} from run directory name {text!r}. "
            "Either edit infer_run_parameters_from_name() or use a manifest file."
        )
    return float(match.group(1))


# =============================================================================
# Run discovery and run-parameter parsing
# =============================================================================

def infer_run_parameters_from_name(run_dir: Path) -> Dict[str, float]:
    """
    Infer logM, logfedd, and alpha from a run-directory name.

    Edit the regular expressions below to match your directory names.

    Examples that this function can be adapted to parse:
        logM_6.0_logfedd_-2_alpha_0.01
        M6.0_fEdd-2_alpha0.01
        mSMBH_6.5_lambda_-1_alpha_0.1

    The function should return:
        {
            "logM": float,
            "logfedd": float,
            "alpha": float,
        }
    """

    name = run_dir.name

    # edit these regexes if your folder names differ.
    logM = safe_float_from_regex(
        r"(?:logM|M|mSMBH|MSMBH)[_=]?(-?\d+(?:\.\d+)?)",
        name,
        "logM",
    )

    fedd = safe_float_from_regex(
        r"(?:logfedd|logfEdd|fedd|fEdd|lambda)[_=]?(-?\d+(?:\.\d+)?)",
        name,
        "fEdd",
    )

    alpha = safe_float_from_regex(
        r"(?:alpha|a)[_=]?(\d+(?:\.\d+)?)",
        name,
        "alpha",
    )

    return {
        "logM": logM,
        "logfedd": np.log10(fedd),
        "alpha": alpha,
    }

def discover_runs(runs_roots=None, grid_alphas=None, run_labels=None) -> List[Dict]:
    """
    Discover AGN simulation runs using the nested directory structure:

        RUNS_ROOT / disk / alpha_{alpha} / logM_{Mbh} / fEdd_{fEdd} /
            {torque}-3bb_{sigma}-{gh}-{progr}-tau_x_{lifetime} / outputs

    Returns
    -------
    runs : list of dict
        Each dict contains the run path and metadata needed downstream.
    """

    if runs_roots is None:
        runs_roots = RUNS_ROOTS

    if grid_alphas is None:
        grid_alphas = GRID_ALPHAS_DEFAULT

    if run_labels is None:
        run_labels = [RUN_LABEL_DEFAULT]
    elif isinstance(run_labels, str):
        run_labels = [run_labels]
    else:
        run_labels = list(run_labels)

    # Keep alpha values as strings because they are used in directory names,
    # e.g. alpha_0.1 and alpha_0.01.
    grid_alphas = [str(a) for a in grid_alphas]

    # Allow either a single path or a list of paths.
    if isinstance(runs_roots, (str, Path)):
        runs_roots = [runs_roots]

    runs = []

    for root in runs_roots:
        root = Path(root)

        if not root.exists():
            print(f"[discover_runs] Skipping missing root: {root}")
            continue

        for disk in GRID_DISKS:
            for alpha in grid_alphas:
                for Mbh in GRID_LOGM_BH:
                    for fEdd in GRID_FEDD:
                        for label in run_labels:

                            run_dir = (
                                root
                                / disk
                                / f"alpha_{alpha}"
                                / f"logM_{Mbh}"
                                / f"fEdd_{fEdd}"
                                / f"{label}"
                                / "outputs"
                            )

                            first_file = run_dir / "first_gen.txt"
                            nth_file = run_dir / "nth_generation.txt"

                            # Keep only combinations where at least one catalog exists.
                            if not first_file.exists() and not nth_file.exists():
                                continue

                            fEdd_float = float(fEdd)

                            runs.append(
                                {
                                    "run_dir": run_dir,

                                    # Core coordinates used by the cosmological weights.
                                    "logM": float(Mbh),
                                    "f_Edd": fEdd_float,
                                    "logfedd": np.log10(fEdd_float),
                                    "alpha": float(alpha),

                                    # Metadata/provenance.
                                    "disk": disk,
                                    "label": label,

                                    # Optional file paths.
                                    "first_file": first_file,
                                    "nth_file": nth_file,
                                }
                            )

    print(f"[discover_runs] Found {len(runs)} runs with at least one merger file.")

    return runs


def read_run_manifest(manifest_path: Path) -> List[Dict]:
    """
    Read a run manifest.

    #HERE:
    If you prefer, create a TSV/CSV file with one row per run.

    Required columns:
        run_dir
        logM
        logfedd
        alpha

    Optional columns:
        N_sample
        N_BH
        run_weight

    Example:

        run_dir                                      logM   logfedd  alpha  N_sample
        /path/to/run/logM_6.0_logfedd_-2_alpha_0.01  6.0   -2.0     0.01   10000

    """

    if manifest_path.suffix.lower() == ".csv":
        df = pd.read_csv(manifest_path)
    else:
        df = pd.read_csv(manifest_path, sep=r"\s+", comment="#")

    required = {"run_dir", "logM", "logfedd", "alpha"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}. "
            f"Available columns are: {list(df.columns)}"
        )

    runs: List[Dict] = []
    for _, row in df.iterrows():
        run = {
            "run_dir": Path(row["run_dir"]),
            "logM": float(row["logM"]),
            "logfedd": float(row["logfedd"]),
            "alpha": float(row["alpha"]),
            "N_sample": float(row.get("N_sample", DEFAULT_N_SAMPLE)),
        }

        if "N_BH" in df.columns and pd.notna(row.get("N_BH")):
            run["N_BH"] = float(row["N_BH"])

        if "run_weight" in df.columns and pd.notna(row.get("run_weight")):
            run["run_weight"] = float(row["run_weight"])

        runs.append(run)

    return runs


# =============================================================================
# Cosmology helper functions
# =============================================================================

def build_cosmology_interpolators(
    zmax: float,
    nz: int = 50000,
) -> Tuple[Callable, Callable]:
    """
    Build interpolators:
        z_to_age(z) -> cosmic age in Gyr
        age_to_z(age_Gyr) -> redshift

    The age-to-redshift mapping is used to assign z_merg from
    t_merg = t_form + t_delay.
    """

    z_grid = np.linspace(0.0, zmax, nz)
    age_grid = cosmo.age(z_grid).value  # Gyr

    # age decreases with z. Sort for interpolation age -> z.
    idx = np.argsort(age_grid)

    age_to_z = interp1d(
        age_grid[idx],
        z_grid[idx],
        bounds_error=False,
        fill_value=np.nan,
    )

    z_to_age = interp1d(
        z_grid,
        age_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    return z_to_age, age_to_z


# =============================================================================
# Cosmological input-file readers
# =============================================================================

class CosmologicalWeights:
    """
    Container for nAGN(z), p(M|z), and p(fEdd|M,z).

    This implementation assumes .npz files, but the load_* methods are
    intentionally isolated so that you can adapt them to your own files.
    """

    def __init__(
        self,
        pM_file: Path,
        pfedd_file: Path,
        nagn_file: Path,
        pfedd_uses_logfedd: bool = True,
    ):
        self.pfedd_uses_logfedd = pfedd_uses_logfedd

        self._load_nagn(nagn_file)
        self._load_pM_given_z(pM_file)
        self._load_pfedd_given_M_z(pfedd_file)

    def _load_nagn(self, nagn_file: Path) -> None:
        """
        Load n_AGN(z) from a simple two-column text file.

        Expected format
        ---------------
        # redshift, nAGN [cMpc^-3]
        0.0   1.23e-4
        0.1   1.20e-4
        ...

        The file may be whitespace-separated or comma-separated.
        Lines beginning with '#' are ignored.

        Returns
        -------
        None
            Sets self.z_nagn, self.nagn_values, and self.nagn_interp.
        """

        path = Path(nagn_file)

        if not path.exists():
            raise FileNotFoundError(f"nAGN file not found: {path}")

        try:
            data = np.loadtxt(path, comments="#", delimiter=None)
        except ValueError:
            # Fallback for comma-separated files.
            data = np.loadtxt(path, comments="#", delimiter=",")

        if data.ndim == 1:
            if data.size != 2:
                raise ValueError(
                    f"Expected two columns in nAGN file {path}, "
                    f"but found one row with {data.size} entries."
                )
            data = data.reshape(1, 2)

        if data.shape[1] < 2:
            raise ValueError(
                f"Expected at least two columns in nAGN file {path}: "
                "redshift and nAGN [cMpc^-3]."
            )

        z = np.asarray(data[:, 0], dtype=float)
        nagn = np.asarray(data[:, 1], dtype=float)

        good = np.isfinite(z) & np.isfinite(nagn) & (nagn >= 0)

        if not np.any(good):
            raise ValueError(f"No valid nAGN rows found in {path}.")

        z = z[good]
        nagn = nagn[good]

        order = np.argsort(z)
        z = z[order]
        nagn = nagn[order]

        # Remove duplicate redshift values, keeping the first occurrence.
        z_unique, unique_idx = np.unique(z, return_index=True)
        nagn_unique = nagn[unique_idx]

        self.z_nagn = z_unique
        self.nagn_values = nagn_unique

        self.nagn_interp = interp1d(
            self.z_nagn,
            self.nagn_values,
            bounds_error=False,
            fill_value=0.0,
        )

    def _load_pM_given_z(self, path: Path) -> None:
        """
        Load p(logM | z).

        Shape convention:
            pM_given_z.shape == (len(z), len(logM))

        """

        data = np.load(path)

        # edit keys if your file uses different names.
        z = data["z_grid"]
        logM = data["M_grid"]
        p = data["pM_given_z"]


        if p.shape != (len(z), len(logM)):
            raise ValueError(
                "pM_given_z has unexpected shape. Expected "
                f"{(len(z), len(logM))}, got {p.shape}. "
                "Edit _load_pM_given_z()."
            )

        self.pM_interp = RegularGridInterpolator(
            (z, logM),
            p,
            bounds_error=False,
            fill_value=0.0,
        )

    def _load_pfedd_given_M_z(self, path: Path) -> None:
        """
        Load p(logfEdd | logM, z), or p(fEdd | logM, z).


        Shape convention:
            pfedd_given_M_z.shape == (len(z), len(logM), len(logfedd))
        """

        data = np.load(path)

        # edit keys if your file uses different names.
        z = data["z_grid"]
        logM = data["M_grid"]

        if self.pfedd_uses_logfedd:
            fedd_axis = data["logfedd"]
        else:
            # edit this key if your file uses e.g. "f_Edd" or "lambda".
            fedd_axis = data["fEdd_grid"]

        p = data["pfedd_given_Mz"]


        if p.shape != (len(z), len(logM), len(fedd_axis)):
            raise ValueError(
                "pfedd_given_M_z has unexpected shape. Expected "
                f"{(len(z), len(logM), len(fedd_axis))}, got {p.shape}. "
                "Edit _load_pfedd_given_M_z()."
            )

        self.pfedd_interp = RegularGridInterpolator(
            (z, logM, fedd_axis),
            p,
            bounds_error=False,
            fill_value=0.0,
        )

    def nAGN(self, z: np.ndarray | float) -> np.ndarray:
        return np.asarray(self.nagn_interp(z))

    def p_M_given_z(self, logM: float, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        points = np.column_stack([z, np.full_like(z, logM, dtype=float)])
        return np.asarray(self.pM_interp(points))

    def p_fedd_given_M_z(
        self,
        logfedd: float,
        logM: float,
        z: np.ndarray,
    ) -> np.ndarray:
        z = np.asarray(z)

        if self.pfedd_uses_logfedd:
            fedd_eval = logfedd
        else:
            fedd_eval = 10.0**logfedd

        points = np.column_stack(
            [
                z,
                np.full_like(z, logM, dtype=float),
                np.full_like(z, fedd_eval, dtype=float),
            ]
        )

        return np.asarray(self.pfedd_interp(points))

    def environment_weight(
        self,
        z_form: np.ndarray,
        logM: float,
        logfedd: float,
    ) -> np.ndarray:
        """
        Intrinsic environment weight evaluated on a formation-time grid.

        Because the redshift-bin builder samples uniformly in cosmic formation
        time, this does NOT include |dt/dz|. The quadrature factor dt_form is
        multiplied separately.

        Weight:
            nAGN(z_form)
            * p(logM | z_form)
            * p(logfEdd | logM, z_form)
        """

        z_form = np.asarray(z_form)

        return (
            self.nAGN(z_form)
            * self.p_M_given_z(logM, z_form)
            * self.p_fedd_given_M_z(logfedd, logM, z_form)
        )


# =============================================================================
# Reading and preparing BBH event catalogs
# =============================================================================

def read_single_file(path: Path) -> pd.DataFrame:
    """
    Read one BBH merger catalog.
    """

    return pd.read_csv(path, sep=INPUT_SEP, engine="python", comment="#")

def _clean_catalog_df(df):
    """
    Remove fully empty rows/columns before concatenation.
    This avoids pandas FutureWarning when one input file is empty/all-NA.
    """
    if df is None or df.empty:
        return None

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return None

    return df


def read_run_catalog(run_dir: Path) -> pd.DataFrame:
    """
    Read first_gen.txt and nth_generation.txt from one run directory,
    concatenate them, and add provenance columns.
    """

    run_dir = Path(run_dir)

    dfs = []
    event_offset = 0

    first_path = run_dir / "first_gen.txt"
    nth_path = run_dir / "nth_generation.txt"

    if first_path.exists() and first_path.stat().st_size > 0:
        df_first = pd.read_csv(first_path, sep=r"\s+", comment="#")
        df_first = _clean_catalog_df(df_first)

        if df_first is not None:
            df_first["origin_file"] = "first_gen"
            df_first["gen_flag"] = 1
            df_first["event_id_within_file"] = np.arange(len(df_first))
            df_first["event_id_within_run"] = np.arange(
                event_offset, event_offset + len(df_first)
            )
            event_offset += len(df_first)
            dfs.append(df_first)

    if nth_path.exists() and nth_path.stat().st_size > 0:
        df_nth = pd.read_csv(nth_path, sep=r"\s+", comment="#")
        df_nth = _clean_catalog_df(df_nth)

        if df_nth is not None:
            df_nth["origin_file"] = "nth_generation"
            df_nth["gen_flag"] = 2
            df_nth["event_id_within_file"] = np.arange(len(df_nth))
            df_nth["event_id_within_run"] = np.arange(
                event_offset, event_offset + len(df_nth)
            )
            event_offset += len(df_nth)
            dfs.append(df_nth)

    if len(dfs) == 0:
        return pd.DataFrame()

    # Keep only genuinely non-empty frames before concatenating.
    dfs = [df for df in dfs if df is not None and not df.empty]

    if len(dfs) == 0:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True, sort=False)

    return df


def get_run_weight(run: Dict) -> float:
    """
    Return the run-level weight.

    For an intrinsic event-property catalog, a common choice is:

        run_weight = N_BH / N_sample

    or simply

        run_weight = 1 / N_sample

    depending on whether you want AGNs with more embedded BHs to contribute more.

    Edit this depending on your normalization.
    """

    return 1.0 


# =============================================================================
# Redshift-bin catalog construction
# =============================================================================
def build_catalog_for_zbin(
    runs: Sequence[Dict],
    zbin: Tuple[float, float],
    output_path: Path,
    weights: CosmologicalWeights,
    z_to_age: Callable,
    age_to_z: Callable,
    z_form_min: float = Z_FORM_MIN,
    z_form_max: float = Z_FORM_MAX,
    nz_form_per_event: int = NZ_FORM_PER_EVENT,
    t_delay_col: str = T_DELAY_COL,
    t_delay_unit: str = T_DELAY_UNIT,
    write_empty_files: bool = True,
) -> pd.DataFrame:
    """
    Build one intrinsic weighted BBH catalog for one merger-redshift bin.

    This version is adapted to the run-directory structure:

        RUNS_ROOT / disk / alpha_{alpha} / logM_{Mbh} / fEdd_{fEdd} /
            {torque}-3bb_{sigma}-{gh}-{progr}-tau_x_{lifetime} / outputs

    Each run dictionary is expected to contain at least:

        run["run_dir"]
        run["logM"]
        run["fedd"]
        run["alpha"]

    and may also contain:

        run["disk"]
        run["f_Edd"]
        run["torque"]
        run["sigma"]
        run["progr"]
        run["pairup"]
        run["gas_hardening"]
        run["lifetime"]
    """

    z_low, z_high = zbin

    if z_high <= z_low:
        raise ValueError(f"Invalid redshift bin {zbin}. Need z_high > z_low.")

    # Cosmic age interval for merger redshifts in [z_low, z_high).
    # Age decreases with z.
    t_merg_min = float(z_to_age(z_high))  # Gyr, earlier cosmic time
    t_merg_max = float(z_to_age(z_low))   # Gyr, later cosmic time

    if not np.isfinite(t_merg_min) or not np.isfinite(t_merg_max):
        raise ValueError(
            f"Could not evaluate cosmic ages for merger bin {zbin}. "
            "Increase cosmology interpolation zmax."
        )

    # Global allowed formation-age interval.
    t_form_global_min = float(z_to_age(z_form_max))  # earliest age allowed
    t_form_global_max = float(z_to_age(z_form_min))  # latest age allowed

    rows: List[Dict] = []

    for run_index, run in enumerate(runs):
        run_dir = Path(run["run_dir"])

        # Core grid coordinates.
        logM = float(run["logM"])
        logfedd = float(run["logfedd"])
        alpha = float(run["alpha"])

        # Metadata from your nested run structure.
        # These are optional so the function does not crash if a run dict
        # is missing one of them.
        disk = run.get("disk", np.nan)
        f_Edd = float(run.get("f_Edd", 10.0**logfedd))
        torque = run.get("torque", "")
        sigma = run.get("sigma", np.nan)
        progr = run.get("progr", "")
        pairup = run.get("pairup", "")
        gas_hardening = run.get("gas_hardening", run.get("gh", ""))
        lifetime = run.get("lifetime", np.nan)

        run_weight = get_run_weight(run)

        df = read_run_catalog(run_dir)

        if df.empty:
            continue

        if t_delay_col not in df.columns:
            raise ValueError(
                f"Delay-time column {t_delay_col!r} not found in {run_dir}. "
                f"Available columns are: {list(df.columns)}"
            )

        # Convert delay column once.
        t_delay_gyr_all = convert_delay_to_gyr(
            df[t_delay_col].to_numpy(dtype=float),
            t_delay_unit,
        )

        # Important: df.iterrows() preserves the dataframe index, which may not
        # be 0, 1, 2, ... after concatenating first_gen and nth_generation.
        # So use enumerate() to index t_delay_gyr_all safely.
        for row_pos, (_, ev) in enumerate(df.iterrows()):
            t_delay_gyr = float(t_delay_gyr_all[row_pos])

            if not np.isfinite(t_delay_gyr):
                continue

            if t_delay_gyr < 0.0:
                continue

            # Allowed formation-time interval for this event to merge in this bin.
            t_form_min = t_merg_min - t_delay_gyr
            t_form_max = t_merg_max - t_delay_gyr

            # If even the latest possible formation time is before the Big Bang, skip.
            if t_form_max <= 0.0:
                continue

            # Intersect with the global formation-redshift range.
            t_form_min = max(t_form_min, t_form_global_min)
            t_form_max = min(t_form_max, t_form_global_max)

            if t_form_min >= t_form_max:
                continue

            # Deterministic quadrature points uniform in cosmic formation time.
            if nz_form_per_event == 1:
                t_form_grid = np.array([0.5 * (t_form_min + t_form_max)])
                dt_form = t_form_max - t_form_min
            else:
                # Midpoint rule avoids giving special weight to bin edges.
                edges = np.linspace(t_form_min, t_form_max, nz_form_per_event + 1)
                t_form_grid = 0.5 * (edges[:-1] + edges[1:])
                dt_form = (t_form_max - t_form_min) / nz_form_per_event

            z_form_grid = age_to_z(t_form_grid)

            good = np.isfinite(z_form_grid)
            if not np.any(good):
                continue

            t_form_grid = t_form_grid[good]
            z_form_grid = z_form_grid[good]

            # Forward map to merger redshift.
            t_merg_grid = t_form_grid + t_delay_gyr
            z_merg_grid = age_to_z(t_merg_grid)

            good = (
                np.isfinite(z_merg_grid)
                & (z_merg_grid >= z_low)
                & (z_merg_grid < z_high)
                & (z_form_grid >= z_form_min)
                & (z_form_grid <= z_form_max)
            )

            if not np.any(good):
                continue

            z_form_grid = z_form_grid[good]
            z_merg_grid = z_merg_grid[good]

            env_w = weights.environment_weight(z_form_grid, logM, logfedd)
            weight_raw = run_weight * env_w * dt_form

            finite_positive = np.isfinite(weight_raw) & (weight_raw > 0.0)

            if not np.any(finite_positive):
                continue

            z_form_grid = z_form_grid[finite_positive]
            z_merg_grid = z_merg_grid[finite_positive]
            weight_raw = weight_raw[finite_positive]

            # Convert event row once to dict.
            ev_dict = ev.to_dict()

            for zf, zm, w in zip(z_form_grid, z_merg_grid, weight_raw):
                out = dict(ev_dict)

                # Unique identifiers and provenance.
                #out["global_run_index"] = run_index
                out["run_dir"] = str(run_dir)

                #out["event_uid"] = (
                #    f"run{run_index:05d}_"
                #    f"event{int(ev_dict['event_id_within_run']):08d}_"
                #    f"zf{zf:.6f}"
                #)

                # Core AGN parameters.
                out["disk"] = disk
                out["logM_SMBH"] = logM
                #out["logf_Edd"] = logfedd
                out["f_Edd"] = f_Edd
                out["alpha"] = alpha

                # Extra parameters from your run-dir structure.
                #out["torque"] = torque
                #out["sigma"] = sigma
                #out["progr"] = progr
                #out["pairup"] = pairup
                #out["gas_hardening"] = gas_hardening
                #out["lifetime"] = lifetime

                # Assigned redshifts.
                out["z_form"] = float(zf)
                out["z_merg"] = float(zm)

                # Delay in standardized units.
                out["t_delay_Gyr"] = t_delay_gyr
                out["t_delay_Myr_standard"] = t_delay_gyr * 1.0e3

                # Weights.
                out["weight_raw"] = float(w)

                rows.append(out)

    out_df = pd.DataFrame(rows)

    if len(out_df) > 0:
        total_weight = out_df["weight_raw"].sum()

        if total_weight > 0.0 and np.isfinite(total_weight):
            out_df["weight_norm"] = out_df["weight_raw"] / total_weight
        else:
            out_df["weight_norm"] = np.nan

    else:
        # Predictable columns for empty files.
        out_df = pd.DataFrame(
            columns=[
                "event_uid",
                "global_run_index",
                "run_dir",
                "event_id_within_run",
                "origin_file",
                "gen_flag",
                "disk",
                "logM_SMBH",
                "logf_Edd",
                "f_Edd",
                "alpha",
                "torque",
                "sigma",
                "progr",
                "pairup",
                "gas_hardening",
                "lifetime",
                "z_form",
                "z_merg",
                "t_delay_Gyr",
                "t_delay_Myr_standard",
                "weight_raw",
                "weight_norm",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(out_df) > 0 or write_empty_files:
        out_df.to_csv(output_path, sep=OUTPUT_SEP, index=False)

    return out_df

def build_all_redshift_catalogs(
    runs: Sequence[Dict],
    z_bins: Sequence[Tuple[float, float]],
    output_root: Path,
    weights: CosmologicalWeights,
    z_form_min: float,
    z_form_max: float,
    nz_form_per_event: int,
    t_delay_col: str,
    t_delay_unit: str,
) -> None:
    zmax_interp = max(
        z_form_max,
        max(z_high for _, z_high in z_bins),
    ) + 1.0

    z_to_age, age_to_z = build_cosmology_interpolators(
        zmax=zmax_interp,
        nz=50000,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for z_low, z_high in z_bins:
        zdir = output_root / format_zbin_dir(z_low, z_high)
        output_path = zdir / "BBH_mergers.txt"

        print(f"\nBuilding intrinsic catalog for z_merg in [{z_low}, {z_high})")

        df_bin = build_catalog_for_zbin(
            runs=runs,
            zbin=(z_low, z_high),
            output_path=output_path,
            weights=weights,
            z_to_age=z_to_age,
            age_to_z=age_to_z,
            z_form_min=z_form_min,
            z_form_max=z_form_max,
            nz_form_per_event=nz_form_per_event,
            t_delay_col=t_delay_col,
            t_delay_unit=t_delay_unit,
        )

        n_rows = len(df_bin)
        raw_weight_sum = (
            float(df_bin["weight_raw"].sum())
            if n_rows > 0 and "weight_raw" in df_bin.columns
            else 0.0
        )

        print(f"  wrote {n_rows} weighted event copies")
        print(f"  output: {output_path}")
        print(f"  sum(weight_raw): {raw_weight_sum:.6e}")

        summary_rows.append(
            {
                "z_low": z_low,
                "z_high": z_high,
                "n_weighted_event_copies": n_rows,
                "sum_weight_raw": raw_weight_sum,
                "output_path": str(output_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_root / "catalog_summary.txt"
    summary.to_csv(summary_path, sep="\t", index=False)

    print(f"\nWrote summary to {summary_path}")


# =============================================================================
# Command-line interface
# =============================================================================

def parse_z_bins(z_bins_string: Optional[str]) -> List[Tuple[float, float]]:
    """
    Parse redshift bins from a string like:
        "0,0.5;0.5,1;1,2;2,4"

    If None, use Z_MERG_BINS.
    """

    if z_bins_string is None:
        return list(Z_MERG_BINS)

    bins = []
    for piece in z_bins_string.split(";"):
        left, right = piece.split(",")
        bins.append((float(left), float(right)))

    return bins


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build intrinsic BBH event-property catalogs in merger-redshift bins."
        )
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        nargs="+",
        default=RUNS_ROOTS,
        help=(
            "Root directory/directories containing the nested AGN grid. "
            "Example: --runs-root /path/to/RUNS_new_spin"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional run manifest with columns run_dir, logM, logfedd, alpha. "
            "If provided, this overrides automatic run discovery."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for output redshift-bin catalogs.",
    )

    parser.add_argument(
        "--pM-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit input file containing p(logM | z). "
            "If omitted, the file is inferred from --redshift-model."
        ),
    )

    parser.add_argument(
        "--pfedd-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit input file containing p(logfEdd | logM, z). "
            "If omitted, the file is inferred from --redshift-model."
        ),
    )

    parser.add_argument(
        "--nagn-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit input file containing nAGN(z). "
            "If omitted, the file is inferred from --redshift-model and "
            "--agn-abundance."
        ),
    )

    parser.add_argument(
        "--pfedd-uses-logfedd",
        type=parse_bool,
        default=PFEDD_USES_LOGFEDD,
        help="Whether the pfedd axis is log10(fEdd).",
    )

    parser.add_argument(
        "--z-form-min",
        type=float,
        default=Z_FORM_MIN,
        help="Minimum AGN formation redshift.",
    )

    parser.add_argument(
        "--z-form-max",
        type=float,
        default=Z_FORM_MAX,
        help="Maximum AGN formation redshift.",
    )

    parser.add_argument(
        "--nz-form-per-event",
        type=int,
        default=NZ_FORM_PER_EVENT,
        help="Number of formation-time quadrature points per event per z bin.",
    )

    parser.add_argument(
        "--t-delay-col",
        type=str,
        default=T_DELAY_COL,
        help="Delay-time column in the BBH merger catalogs.",
    )

    parser.add_argument(
        "--t-delay-unit",
        type=str,
        default=T_DELAY_UNIT,
        choices=["yr", "Myr", "Gyr"],
        help="Unit of the delay-time column.",
    )

    parser.add_argument(
        "--z-bins",
        type=str,
        default=None,
        help=(
            "Optional redshift bins, e.g. '0,0.5;0.5,1;1,2'. "
            "If omitted, uses Z_MERG_BINS from the script."
        ),
    )

    parser.add_argument(
        "--alpha",
        "--alphas",
        dest="alphas",
        nargs="+",
        default=GRID_ALPHAS_DEFAULT,
        help=(
            "Disk alpha values to include. "
            "Examples: --alpha 0.1 or --alphas 0.1 0.01"
        ),
    )

    parser.add_argument(
        "--redshift-model",
        default=REDSHIFT_MODEL_DEFAULT,
        choices=["EL", "SE"],
        help="Redshift model to use, e.g. EL or SE.",
    )

    parser.add_argument(
        "--agn-abundance",
        default=AGN_ABUNDANCE_DEFAULT,
        choices=["LAM", "HAM"],
        help="AGN abundance model to use, e.g. LAM or HAM.",
    )

    parser.add_argument(
        "--run-label",
        "--run-labels",
        dest="run_label",
        nargs="+",
        default=[RUN_LABEL_DEFAULT],
        help=(
            "Run-directory label(s) to include. Example: "
            "--run-label G24_K18-3bb_0.0-IG25-agnostic-tau_x_1."
        ),
    )

    args = parser.parse_args()

    # Keep alpha as strings for directory construction: alpha_0.1, alpha_0.01.
    selected_alphas = [str(a) for a in args.alphas]
    redshift_model = args.redshift_model
    agn_abundance = args.agn_abundance

    # ------------------------------------------------------------------
    # Cosmological input files
    # ------------------------------------------------------------------
    pm_file = (
        args.pM_file
        if args.pM_file is not None
        else Path(f"../input/population_models/pM_given_z_{redshift_model}.npz")
    )

    pfedd_file = (
        args.pfedd_file
        if args.pfedd_file is not None
        else Path(f"../input/population_models/pfEdd_given_Mz_{redshift_model}.npz")
    )

    nagn_file = (
        args.nagn_file
        if args.nagn_file is not None
        else Path(
            f"../input/population_models/nAGN_models/"
            f"nAGN_{redshift_model}_{agn_abundance}.txt"
        )
    )

    # ------------------------------------------------------------------
    # Redshift bins
    # ------------------------------------------------------------------
    if args.z_bins is None:
        z_bins = Z_MERG_BINS
    else:
        z_bins = parse_z_bins(args.z_bins)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    # This avoids overwriting different cosmology / alpha choices.
    output_root = (
        Path(args.output_root)
        / f"redshift_model_{redshift_model}"
        / f"agn_abundance_{agn_abundance}"
        / f"alpha_{'_'.join(selected_alphas)}"
    )

    print("[main] Configuration")
    print(f"  selected alphas:        {selected_alphas}")
    print(f"  redshift model:         {redshift_model}")
    print(f"  AGN abundance model:    {agn_abundance}")
    print(f"  p(M|z) file:            {pm_file}")
    print(f"  p(fEdd|M,z) file:       {pfedd_file}")
    print(f"  nAGN(z) file:           {nagn_file}")
    print(f"  runs root(s):           {args.runs_root}")
    print(f"  output root:            {output_root}")
    print(f"  z formation range:      {args.z_form_min} <= z_form <= {args.z_form_max}")
    print(f"  nz form per event:      {args.nz_form_per_event}")
    print(f"  delay column/unit:      {args.t_delay_col} [{args.t_delay_unit}]")
    print(f"  redshift bins:          {z_bins}")

    for label, path in [
        ("p(M|z)", pm_file),
        ("p(fEdd|M,z)", pfedd_file),
        ("nAGN(z)", nagn_file),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    # ------------------------------------------------------------------
    # Load cosmological weights
    # ------------------------------------------------------------------
    weights = CosmologicalWeights(
        pM_file=pm_file,
        pfedd_file=pfedd_file,
        nagn_file=nagn_file,
        pfedd_uses_logfedd=args.pfedd_uses_logfedd,
    )

    # ------------------------------------------------------------------
    # Discover or load runs
    # ------------------------------------------------------------------
    if args.manifest is not None:
        runs = read_run_manifest(args.manifest)
    else:
        runs = discover_runs(
            runs_roots=args.runs_root,
            grid_alphas=selected_alphas,
            run_labels=args.run_label,
        )

    # ------------------------------------------------------------------
    # Diagnostic: check which AGN grid points are actually included
    # ------------------------------------------------------------------
    runs_df = pd.DataFrame([
        {
            "logM": r["logM"],
            "f_Edd": r.get("f_Edd", 10.0**r["logfedd"]),
            "logfedd": r["logfedd"],
            "alpha": r["alpha"],
            "disk": r.get("disk", ""),
            "label": r.get("label", ""),
            "run_dir": str(r["run_dir"]),
        }
        for r in runs
    ])
    
    print("\n[diagnostic] Runs included in cosmological mixture")
    print(f"  total number of runs: {len(runs_df)}")
    
    print("\n[diagnostic] alpha values:")
    print(np.sort(runs_df["alpha"].unique()))
    
    print("\n[diagnostic] logM grid:")
    print(np.sort(runs_df["logM"].unique()))
    
    print("\n[diagnostic] f_Edd grid:")
    print(np.sort(runs_df["f_Edd"].unique()))
    
    print("\n[diagnostic] number of runs per alpha:")
    print(runs_df.groupby("alpha").size())
    
    print("\n[diagnostic] number of f_Edd values per alpha and logM:")
    coverage = (
        runs_df
        .groupby(["alpha", "logM"])["f_Edd"]
        .nunique()
        .reset_index(name="n_fEdd")
    )
    print(coverage.to_string(index=False))
    
    print("\n[diagnostic] number of logM values per alpha and f_Edd:")
    coverage2 = (
        runs_df
        .groupby(["alpha", "f_Edd"])["logM"]
        .nunique()
        .reset_index(name="n_logM")
    )
    print(coverage2.to_string(index=False))
    
    if len(runs) == 0:
        raise RuntimeError(
            "No runs found. Check --runs-root, selected --alpha values, "
            "and the GRID_* settings."
        )

    print(f"[main] Number of runs: {len(runs)}")

    # ------------------------------------------------------------------
    # Cosmology interpolation
    # ------------------------------------------------------------------
    zmax_interp = max(
        args.z_form_max,
        max(z_high for _, z_high in z_bins),
    ) + 1.0

    z_to_age, age_to_z = build_cosmology_interpolators(
        zmax=zmax_interp,
        nz=30000,
    )

    # ------------------------------------------------------------------
    # Build catalogs
    # ------------------------------------------------------------------
    for zbin in z_bins:
        z_low, z_high = zbin

        zdir = output_root / f"z_{z_low:.1f}_{z_high:.1f}"

        label_string = "_".join(args.run_label)
        label_string = label_string.replace("/", "_")
        label_string = label_string.rstrip(".")

        output_path = zdir / f"BBH_mergers_{label_string}.txt"

        print(f"[main] Building catalog for z_merg in [{z_low}, {z_high})")
        print(f"       output: {output_path}")

        df_bin = build_catalog_for_zbin(
            runs=runs,
            zbin=zbin,
            output_path=output_path,
            weights=weights,
            z_to_age=z_to_age,
            age_to_z=age_to_z,
            z_form_min=args.z_form_min,
            z_form_max=args.z_form_max,
            nz_form_per_event=args.nz_form_per_event,
            t_delay_col=args.t_delay_col,
            t_delay_unit=args.t_delay_unit,
            write_empty_files=True,
        )

        print(
            f"[main] Finished z=[{z_low}, {z_high}): "
            f"{len(df_bin)} weighted event copies"
        )

    print("[main] Done.")



if __name__ == "__main__":
    main()
