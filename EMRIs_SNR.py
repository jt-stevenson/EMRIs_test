import pandas as pd
import numpy as np
import lal
import numpy.random as rng
import matplotlib.pyplot as plt

import warnings
import argparse
import time
import h5py

from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux, get_separatrix
from few import get_file_manager

from scipy.interpolate import CubicSpline

from astropy.cosmology import Planck18 as COSMO
from astropy.cosmology import z_at_value
import astropy.units as u


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

################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM', 'NT'])
    parser.add_argument('-TT', type=str, default="P10", choices=['P10', 'B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-BIMF', type=str, default="PY", choices=['Vaccaro', 'Tagawa', 'Bartos', 'PY'])
    parser.add_argument('-RD', type=str, default="PY", choices=['Bartko', 'Rom', "PY"])
    parser.add_argument('-wind', type=str, default="On", choices=['On', 'Off', "Partial"])
    parser.add_argument('-a', type=float, default=0.1)
    parser.add_argument('-le', type=float, default=0.01)
    parser.add_argument('-spin', type=float, default=0.9)  # real number
    parser.add_argument('-Mbh', type=float, default=1e7)   # MSun
    parser.add_argument('-T', type=float, default=1e7)     # Myrs
    parser.add_argument('-plot', action='store_true')      # truth value
    parser.add_argument('-date', action='store_true')      # truth value
    parser.add_argument('-iter', type=int)
    
    args = parser.parse_args()
    return args
################################################################################################

################################################################################################
### Loading information on mass distribution ###################################################
################################################################################################
if __name__ == '__main__':
    args=main()

    N=6311
    Mbh=args.Mbh

    filename=f'EMRI_Rates/{args.BIMF}/MBH_{args.Mbh}/{args.DT}/alpha_{args.a}/spin_{args.spin}/Tdisk_{args.T}/wind_{args.wind}/EMRIs_{args.TT}_1g_*.txt'

    with open(filename) as f:
        lines = f.readlines()

    header_end = lines.index("Data:\n") + 1

    for i in range (4, header_end-2):
        print(f'{lines[i].split(" = ")[0]}={lines[i].split(" = ")[1]}')

    data = pd.read_csv(filename, delimiter=" ", skiprows=header_end)

    data.columns = [col.strip().replace(",", "") for col in data.columns]
    print(data.keys())

    N=len(data["m1/Msun"])
    print(f'Columns: {header_end}, Rows: {N}')

    m1=data['m1/Msun']
    print(m1[0])


    Tobs = 4  # observation time (years), if the inspiral is shorter, the it will be zero padded
    dt = 5    # time interval (seconds)
    mode_selection_threshold = 1e-4  # relative threshold for mode inclusion: only modes making a relative contribution to
                # the total power above this threshold will be included in the waveform.
    x0 = 1.0 #initial cos(inclination) - fine to assume as 1 due to short timescale of alignment compared to inspiral
    e0 = 0  # eccentricity - assumed circular in runs anyway
    a = args.spin   # dimensionless spin parameter for the primary - will be ignored in Schwarzschild waveform
    ef = 0.0

    SNRs_Speri=[]
    SNR_counts=[]

    for i in range(0, N):
        m2 = data['m1/Msun'][i] # secondary object mass (solar masses)
        snr_speri=compute_snr(m1, m2 , a, Tobs, ef, z, dt, psd='LISA_FEW')

        SNRs_Speri.append(snr_speri)
