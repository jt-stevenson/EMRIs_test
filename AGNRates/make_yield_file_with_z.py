from pathlib import Path

import numpy as np
import pandas as pd
import argparse
import warnings
import h5py

from datetime import datetime, timedelta

from few import get_file_manager
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux, get_separatrix

from scipy.interpolate import CubicSpline

from astropy.cosmology import Planck18 as COSMO

MERGER_FLAG = 7
DELAY_COLUMN = "t_inspiral/Myr"
FLAG_COLUMN = "total_flags"
SNR_COLUMN= "SNRs"
BIN_EDGES_GYR = np.logspace(-9, -1, 80)

td_gen = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    return_list=True,
)

#from lsperi notebooks
warnings.filterwarnings("ignore")
EPS = 1e-2
MODES = [(ll, mm, nn) for ll in range(2, 5) for mm in range(1, ll + 1) for nn in range(-1, 3)]

DEFAULT_ANGLES = {
    "qS": np.pi / 3,
    "phiS": np.pi / 3,
    "qK": np.pi / 3,
    "phiK": np.pi / 3,
    "Phi_phi0": np.pi / 3,
    "Phi_theta0": 0.0,
    "Phi_r0": np.pi / 3,
}

global FEW_GEN, TRAJ, RHS

TRAJ = EMRIInspiral(func=KerrEccEqFlux)
RHS = KerrEccEqFlux()
FEW_GEN = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    return_list=True,
)

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

#from lsperi notebooks
def redshift_to_luminosity_distance(z):
    return COSMO.luminosity_distance(z).value * 1e-3  # Gpc

def get_spline_psd(filename="LISA v1.0_PSD.h5"):
    with h5py.File(filename, "r") as data:
        key = filename.split("_PSD.h5")[0]
        psd_data = data[key]["sensitivities_links"][()]
        f_psd = data[key]["f"][()]
    return CubicSpline(f_psd, psd_data)


def get_spline_psd_alice(filename):
    f_psd, asd_psd = np.loadtxt(filename, unpack=True)
    return CubicSpline(f_psd, asd_psd ** 2)


def get_psd_wrapper(psd="LISA"):
    if psd == "LISA":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foregrounds_LISA_LS_asd.dat", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foregrounds_LISA_LS_asd.dat...", cubic_spline_psd)
        fmin, fmax = 1e-4, 1.0
    elif psd == "AMADEUS":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foreground_AMADEUS-Baseline_asd.txt", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foreground_AMADEUS-Baseline_asd.txt...", cubic_spline_psd)
        fmin, fmax = 1e-6, 1.0
    elif psd == "DO-IT":
        f_psd, asd_psd = np.loadtxt("PSD_plus_foreground_DO-IT-Baseline_asd.txt", unpack=True)
        cubic_spline_psd = CubicSpline(f_psd, asd_psd**2)
        # print(f"Using PSD from PSD_plus_foreground_DO-IT-Baseline_asd.txt...", cubic_spline_psd)
        fmin, fmax = 1e-4, 10.0
    #my addition
    elif psd == 'LISA_FEW':
        data = np.loadtxt(get_file_manager().get_file("LPA.txt"), skiprows=1)
        data[:, 1] = data[:, 1] ** 2
        # define PSD function
        cubic_spline_psd = CubicSpline(*data.T)
        fmin, fmax = 1e-4, 1.0
    return cubic_spline_psd, fmin, fmax


def get_initial_conditions(params, err=1e-6):
    m1, m2, a, Tpl, ef = params
    x0 = 1.0
    RHS.add_fixed_parameters(m1, m2, a)

    p_0 = TRAJ.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, ef, x0) + 1e-3
    forward_result = TRAJ(m1, m2, a, p_0, ef, x0, T=10.0, integrate_backwards=False, err=err)
    backwards_result = TRAJ(
        m1,
        m2,
        a,
        forward_result[1][-1],
        forward_result[2][-1],
        x0,
        T=Tpl,
        integrate_backwards=True,
        err=err,
    )

    p0 = backwards_result[1][-1]
    e0 = backwards_result[2][-1]
    x0 = backwards_result[3][-1]

    f_phi_theta_r = TRAJ.inspiral_generator.eval_integrator_derivative_spline(backwards_result[0], order=1)
    f_phi = -f_phi_theta_r[:, 3] / (2 * np.pi)
    f_r = -f_phi_theta_r[:, 5] / (2 * np.pi)
    return p0, e0, x0, f_phi, f_r

def compute_snr(
    m1,
    m2,
    a,
    Tobs,
    ef,
    z,
    dt,
    qS=None,
    phiS=None,
    qK=None,
    phiK=None,
    Phi_phi0=None,
    Phi_theta0=None,
    Phi_r0=None,
    psd="LISA_FEW",
    num_freq=5000,
):

    qS = DEFAULT_ANGLES["qS"] if qS is None else qS
    phiS = DEFAULT_ANGLES["phiS"] if phiS is None else phiS
    qK = DEFAULT_ANGLES["qK"] if qK is None else qK
    phiK = DEFAULT_ANGLES["phiK"] if phiK is None else phiK
    Phi_phi0 = DEFAULT_ANGLES["Phi_phi0"] if Phi_phi0 is None else Phi_phi0
    Phi_theta0 = DEFAULT_ANGLES["Phi_theta0"] if Phi_theta0 is None else Phi_theta0
    Phi_r0 = DEFAULT_ANGLES["Phi_r0"] if Phi_r0 is None else Phi_r0

    dist = redshift_to_luminosity_distance(z)
    try:
        p0, e0, x0, _, _ = get_initial_conditions(np.asarray([m1 * (1 + z), m2 * (1 + z), a, Tobs, ef]))
    except Exception as exc:
        print(f"Error computing initial conditions for m1={m1}, m2={m2}, a={a}, Tobs={Tobs}, ef={ef}, z={z}: {exc}")
        return 0.0
        
    cubic_spline_psd, fmin, fmax = get_psd_wrapper(psd)
    f_pos = np.linspace(fmin, fmax, num=num_freq)
    freq = np.hstack((-f_pos[::-1], np.asarray([0.0]), f_pos))

    hf = FEW_GEN(
        m1 * (1 + z),
        m2 * (1 + z),
        a,
        p0,
        e0,
        x0,
        dist,
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        T=Tobs,
        dt=dt,
        f_arr=freq,
        mask_positive=True,
        mode_selection=MODES,
    )

    h_plus = np.asarray(hf[0])[1:]
    h_cross = np.asarray(hf[1])[1:]
    df = f_pos[1] - f_pos[0]
    snr_squared = 4.0 * np.sum((np.abs(h_plus) ** 2 + np.abs(h_cross) ** 2) / cubic_spline_psd(f_pos) * df)
    return float(np.sqrt(snr_squared))

def make_yield_file(LABEL, input_file, z, SNR_lim,):
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

    N_SAMPLE=params['N']
    N_BH=params['N']
    M=params['M_SMBH']
    alpha=params['alpha']

    ef=0
    Tobs = 4  # observation time (years), if the inspiral is shorter, the it will be zero padded
    dt = 5.0    # time interval (seconds)
    mode_selection_threshold = 1e-4  # relative threshold for mode inclusion: only modes making a relative contribution to
                # the total power above this threshold will be included in the waveform.
    x0 = 1.0 #initial cos(inclination) - fine to assume as 1 due to short timescale of alignment compared to inspiral
    e0 = 0  # eccentricity - assumed circular in runs anyway

    SNRs=[]

    for i in range(0, len(events)):
        m2=events['m1/Msun,'][i]
        snr=compute_snr(M, m2, 0.9, Tobs, ef, z, dt, psd='LISA_FEW')
        SNRs.append(snr)

    events[SNR_COLUMN] = SNRs

    columns = {column.rstrip(","): column for column in events.columns}

    missing = {DELAY_COLUMN, FLAG_COLUMN} - columns.keys()
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if N_SAMPLE <= 0:
        raise ValueError("N_SAMPLE must be positive")

    is_merger = events[columns[FLAG_COLUMN]] == MERGER_FLAG
    is_detectable = events[columns[SNR_COLUMN]] >= SNR_lim
    is_detectable_merger = is_merger & is_detectable

    print(events.loc[is_detectable_merger])

    delays_gyr = (
            pd.to_numeric(events.loc[is_detectable_merger, columns[DELAY_COLUMN]], errors="coerce")
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

    output_file=Path(f'/Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/SNRs_{SNR_lim}/{LABEL}/M_{M:.1e}_fEdd_0.01/z_{z}/yield_1g.txt')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "t_delay_Gyr": centers,
            "Y_dt_delay_Gyr_given_M_fedd": yields,
        }
    ).to_csv(output_file, sep="\t", index=False)
    
    summary_file = Path(f'/Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/SNRs_{SNR_lim}/{LABEL}/M_{M:.1e}_fEdd_0.01/summary_yields.txt')
    if summary_file.exists():
        file = open(summary_file, "a")
        file.write(f'{z} {LABEL} {N_BH} {int(is_merger.sum())} {int(is_detectable_merger.sum())} {float(np.sum(yields * widths))}\n')
        file.close()
    else:
        pd.DataFrame(
                {"z":[z] , 
                "label": [LABEL], 
                "N_BH": [N_BH], 
                'Nmerg': [int(is_merger.sum())], 
                "Ndetect": [int(is_detectable_merger.sum())], 
                "yield_integral": [float(np.sum(yields * widths))]}
                ).to_csv(f'/Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/SNRs_{SNR_lim}/{LABEL}/M_{M:.1e}_fEdd_0.01/summary_yields.txt',
                sep="\t",
                index=False,
            )

    return {
        "file": str(output_file),
        "summary_file": str(f'/Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/SNRs_{SNR_lim}/{LABEL}/M_{M:.1e}_fEdd_0.01/summary_yields.txt'),
        "label": LABEL,
        "Nsample": N_SAMPLE,
        "Nmerg": int(is_detectable_merger.sum()),
        "N_BH": N_BH,
        "yield_integral": float(np.sum(yields * widths)),
    }

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--infile", required=True, help="input file")
    p.add_argument("--label", required=True, help="Physical-model directory below SG_alpha_<alpha>")
    p.add_argument("--z", required=True, help="redshift")
    p.add_argument("--SNR_lim", required=True, default=30, help="SNR detection limit for LISA")

    return p.parse_args()

def main():
    args = parse_args()
    make_yield_file(LABEL=args.label, input_file=Path(args.infile), z=float(args.z), SNR_lim=float(args.SNR_lim))

if __name__ == "__main__":
    main()