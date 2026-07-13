import matplotlib.pyplot as plt
import pagn.constants as ct
import multiprocessing
import numpy as np
import argparse
import warnings
import pagn
import time
import os
from pagn.opacities import electron_scattering_opacity
from scipy.interpolate import UnivariateSpline
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from datetime import datetime
from tqdm import tqdm

start = datetime.now()
warnings.filterwarnings('ignore')

################################################################################################
### Computation parameters #####################################################################
################################################################################################
plotting = True
printing = False
initialize_radius = 'log_uniform'
##can be "log_uniform" or "capture_probability"

################################################################################################
### Capture timescale ##########################################################################
################################################################################################
def t_damp_hydro(radius, m1, h_ratio, Sigma_gas, M_SMBH):
    #timescale for inclination and eccentricity damping
    m1_arr = np.atleast_1d(m1).astype(float)

    t_damp = np.zeros(m1_arr.size, dtype=float)
    Omega=np.sqrt(ct.G * M_SMBH / radius**3) #s^-1
    #draw initial inclination
    cos_i = np.random.rand(m1_arr.size)
    sin_i = np.sqrt(1.- cos_i**2.)
    i_tilde = sin_i/h_ratio

    ## from Rowan, MNRAS543,132–145(2025)
    ## eq.46, fig. 8
    R_Hill = radius * (m1/M_SMBH/3.)**(1./3.)
    m_tilde = Sigma_gas * np.pi * R_Hill**2. / m1   #ambient Hill mass
    
    i_tilde_c = 4.6
    A = 10.**(0.67 * np.log10(m_tilde) - 2.64 * np.log10(i_tilde_c) + 1.80)
    B = 10.**(0.67 * np.log10(m_tilde) + 1.80)

    #Delta i_tilde / i_tilde
    Phi_high_inclination = B * i_tilde**(-2.64) 
    Phi_low_inclination = A
    Phi = np.where(i_tilde >= i_tilde_c, Phi_high_inclination, Phi_low_inclination)

    #timescale
    t_damp = np.where(i_tilde <=1, 0., (np.pi / Omega) * (1./Phi))
    return t_damp

def capture_processing(args, disk, M, Rmin, Rmax):
    N=len(M)
    r0 = np.zeros(N)
    for i in range(N):
        timescale_capture = t_damp_hydro(disk.R, M[i], disk.h/disk.R, 2*disk.rho*disk.h, disk.Mbh)
        ##cap capture timescale on disk lifetime
        ok = np.where(timescale_capture < args.T*ct.yr)[0]

        ##weights = 1/t_capt
        eps=1e-10
        beta=1
        w = 1. / np.maximum(timescale_capture[ok], eps) ** beta 
        w_sum = w.sum()
        p = w / w_sum

        ##extract random index
        j = np.random.choice(ok, p=p) 
        r0[i] = disk.R[j]
        """
        while (r0[i]==0):
            iter+=1
            r_temp = 10**(np.log10(Rmin) + (np.log10(Rmax) - np.log10(Rmin)) * np.random.rand(1))
            t_capt = np.interp(r_temp, disk.R, timescale_capture)
            if (t_capt < args.T*ct.yr): ### Temporary
                r0[i]=r_temp
                break
        """
    return r0


################################################################################################
### Migration Torques ##########################################################################
################################################################################################
#
# See documentation on:
# https://dariagangardt.github.io/pAGN/examples.html#evolving-agn-disks
#
def gamma_0(q, hr, Sigma, r, Omega):
    gamma_0 = q*q*Sigma*r*r*r*r*Omega*Omega/(hr*hr)
    return gamma_0

def gamma_iso(dSigmadR, dTdR):
    alpha = - dSigmadR
    beta = - dTdR
    gamma_iso = - 0.85 - alpha - 0.9*beta
    return gamma_iso

def gamma_ad(dSigmadR, dTdR):
    alpha = - dSigmadR
    beta = - dTdR
    gamma = 5/3
    xi = beta - (gamma - 1)*alpha
    gamma_ad = - 0.85 - alpha - 1.7*beta + 7.9*xi/gamma
    return gamma_ad/gamma

def dSigmadR(obj):
    Sigma = 2*obj.rho*obj.h # descrete
    rlog10 = np.log10(obj.R)  # descrete
    Sigmalog10 = np.log10(Sigma)  # descrete
    Sigmalog10_spline = UnivariateSpline(rlog10, Sigmalog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dSigmadR_spline =  Sigmalog10_spline.derivative()
    dSigmadR = dSigmadR_spline(rlog10)
    return dSigmadR

def dSigmadR_reduced(obj, Sigma_reduced):
    Sigma = Sigma_reduced # descrete
    rlog10 = np.log10(obj.R)  # descrete
    Sigmalog10 = np.log10(Sigma)  # descrete
    Sigmalog10_spline = UnivariateSpline(rlog10, Sigmalog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dSigmadR_spline =  Sigmalog10_spline.derivative()
    dSigmadR = dSigmadR_spline(rlog10)
    return dSigmadR

def dTdR(obj):
    rlog10 = np.log10(obj.R)  # descrete
    Tlog10 = np.log10(obj.T)  # descrete
    Tlog10_spline = UnivariateSpline(rlog10, Tlog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dTdR_spline = Tlog10_spline.derivative()
    dTdR = dTdR_spline(rlog10)
    return dTdR

def dPdR(obj):
    rlog10 = np.log10(obj.R)  # descrete
    pgas = obj.rho * obj.T * ct.Kb / ct.massU
    prad = obj.tauV*ct.sigmaSB*obj.Teff4/(2*ct.c)
    ptot = pgas + prad
    Plog10 = np.log10(ptot)  # descrete
    Plog10_spline = UnivariateSpline(rlog10, Plog10, k=3, s=0.005, ext=0)  # need scipy ver 1.10.0
    dPdR_spline = Plog10_spline.derivative()
    dPdR = dPdR_spline(rlog10)
    return dPdR
    
# modified by Paola Vaccaro
def Theta(obj):
    c_v = 1.5 * ct.Kb * 6.02214076e23 #specific heat capacity for a monoatomic gas
    Sigma = obj.rho * obj.h
    #Omega=np.sqrt(ct.G * Mbh / obj.R**3) #s^-1
    #print(Omega/obj.Omega)

    kes = electron_scattering_opacity(X=0.7)
    tau = kes * Sigma /2.
    tau_eff = 3.*tau/8. + np.sqrt(3.)/4. + 1./(4.*tau)

    Theta = c_v * Sigma * obj.Omega * tau_eff / 12. / np.pi / ct.sigmaSB / (obj.T**3.)
    return Theta

# modified by Paola Vaccaro
def CI_p10(obj, dSigmadR, dTdR):
    cI = -0.85 + 0.9*dTdR + dSigmadR #Paardekooper et al. 2010 (iso)
    return cI

def CI_b16(obj, dSigmadR, dTdR):
    cI_iso = gamma_iso(dSigmadR, dTdR)
    cI_ad = gamma_ad(dSigmadR, dTdR)

    cI = (cI_ad * Theta(obj)**2. + cI_iso) / (Theta(obj) + 1.)**2.

    return cI

def CI_jm17(dSigmadR, dTdR, gamma, obj):
    cI = (0.46 + 0.96*dSigmadR - 1.8*dTdR)/gamma
    return cI


# modified by Paola Vaccaro
def CL(dSigmadR, dTdR, gamma, obj):
    #lindblad torque
    xi = 16*gamma*(gamma - 1)*ct.sigmaSB*(obj.T*obj.T*obj.T*obj.T)\
        /(3*obj.kappa*obj.rho*obj.rho*obj.h*obj.h*obj.Omega*obj.Omega)
    x2_sqrt = np.sqrt(xi/(2*obj.h*obj.h*obj.Omega))
    fgamma = (x2_sqrt + 1/gamma)/(x2_sqrt+1)
    cL = -(2.34 - 0.1*dSigmadR + 1.5*dTdR)*fgamma
    return cL

def gamma_thermal(gamma, obj, q):
    xi = 16 * gamma * (gamma - 1) * ct.sigmaSB * (obj.T * obj.T * obj.T * obj.T) \
         / (3 * obj.kappa * obj.rho * obj.rho * obj.h * obj.h * obj.Omega * obj.Omega)
    mbh = obj.Mbh*q
    muth = xi * obj.cs / (ct.G * mbh)
    R_Bhalf = ct.G*mbh/obj.cs**2
    muth[obj.h<R_Bhalf] = (xi / (obj.cs*obj.h))[obj.h<R_Bhalf]

    Lc = 4*np.pi*ct.G*mbh*obj.rho*xi/gamma
    lam = np.sqrt(2*xi/(3*gamma*obj.Omega))

    dP = -dPdR(obj)
    xc = dP*obj.h*obj.h/(3*gamma*obj.R)

    kes = electron_scattering_opacity(X=0.7)
    L = 4 * np.pi * ct.G * ct.c * mbh / kes ##this is assuming eddington fraction ==1

    g_hot = 1.61*(gamma - 1)*xc*L/(Lc*gamma*lam) 
    g_cold = -1.61*(gamma - 1)*xc/(gamma*lam)
    g_thermal = g_hot + g_cold
    g_thermal_new = g_hot*(4*muth/(1+4*muth)) + g_cold*(2*muth/(1+2*muth))
    g_thermal[muth < 1] = g_thermal_new[muth < 1]
    decay = 1 - np.exp(-lam*obj.tauV/obj.h)
    return g_thermal*decay *(obj.R/obj.h) 
################################################################################################

################################################################################################
### Our functions ##############################################################################
################################################################################################
def gamma_GW(r, m, M, e=0.0, return_torque=True, use_reduced_mass=True):
    a = r  # identify radius with semimajor axis
    Mtot = M + m
    mu = m*M/Mtot if use_reduced_mass else m

    # Peters (1964) orbit-averaged factors
    one_minus_e2 = max(1.0 - e**2, 1e-12)
    f_e = 1.0 + (73.0/24.0)*e**2 + (37.0/96.0)*e**4
    g_e = 1.0 + (121.0/304.0)*e**2

    # da/dt (Peters 1964)
    da_dt = -(64.0/5.0) * (ct.G**3 * m * M * Mtot) / (ct.c**5 * a**3) * f_e / (one_minus_e2**(7.0/2.0))

    if not return_torque:
        return da_dt  # dr/dt ~ da/dt (negative)

    # de/dt (needed if e>0 for torque through dL/dt)
    de_dt = - (304.0/15.0) * (ct.G**3 * m * M * Mtot) / (ct.c**5 * a**4) * e * g_e / (one_minus_e2**(5.0/2.0))

    # Orbital angular momentum L = μ sqrt(G M a (1 - e^2))
    # dL/dt = μ sqrt(G M) [ (sqrt(1-e^2)/(2 sqrt(a))) da/dt  - (e sqrt(a)/sqrt(1-e^2)) de/dt ]
    sqrtGM = np.sqrt(ct.G*M)
    term_a = (np.sqrt(one_minus_e2) / (2.0*np.sqrt(a))) * da_dt
    term_e = 0.0
    if e > 0.0:
        term_e = (e * np.sqrt(a) / max(np.sqrt(one_minus_e2), 1e-30)) * de_dt
    dL_dt = mu * sqrtGM * (term_a - term_e)

    return dL_dt

def Ledd(MBH, X):
    kappa=0.2 * (1+X)
    Ledd= (4 * np.pi * ct.G * MBH * ct.c) /kappa
    return Ledd

def mdot_damped(m, disk, gamma, wind):
    ##from Zhen Pan, Huan Yang (2021), eq. 36
    ##initially developed by Kocsis, Yunes, Loeb (2011)
    ##combining equations 3-7 from Chen, Ren, Dai (2023)

    ledd=Ledd(m, 0.7)

    Mbh=disk.Mbh
    Mdot=disk.Mdot

    alpha=disk.alpha

    m_edd=ledd/(ct.c**2)

    h_ratio = disk.h / disk.R
    sigma = 2*disk.h*disk.rho
    delta_v_phi = (3. - gamma) / 2. * h_ratio * disk.cs #eq.39a
    delta_v_dr = 1.5 * (m / (3. * Mbh))**(1./ 3.) * h_ratio**(-1.) * disk.cs #eq. 39c

    R_G=Mbh * ct.G /(ct.c*ct.c)

    vgas=-disk.Mdot/(2 * np.pi * disk.R * sigma)
    vstar=-1.3e-6 * (m/(10*ct.MSun))/(Mbh/(1e5*ct.MSun)) * (disk.R/10*R_G)**(-3)

    delta_v_r=np.abs(vgas-vstar)

    vrel2 = (delta_v_phi + delta_v_dr)**2 + delta_v_r**2 

    R_Hill = disk.R * (m / (3. * disk.Mbh))**(1./3.)
    R_BHL = ct.G * m / (vrel2 + disk.cs**2) #Bondi radius

    #eqn 6 from Chen Ren Dai
    r_rel=np.minimum(R_BHL, R_Hill)
    Rd_gap= 0.21 * (m/Mbh)**(1/2) * (disk.h/disk.R)**(-3/4) * alpha**(-1/4) * disk.R

    rho=[]

    for i in range(0, len(r_rel)):
        if r_rel[i]>Rd_gap[i]:
            rho.append(disk.rho[i])
        elif r_rel[i]<Rd_gap[i]:
            rho_d_gap= disk.rho[i] * 1/(1+ (0.04 * (m/Mbh)**2 * (disk.h[i]/disk.R[i])**(-5) * alpha**(-1)))
            rho.append(rho_d_gap)

    rho=np.array(rho)

    mdot_BHL =  (ct.G)**2 * 4.0 * np.pi * rho * m**2. / (vrel2 + disk.cs**2.)**1.5  #Bondi accretion rate, eq.37
    mdot_inflow=mdot_BHL * np.minimum(1., np.minimum(disk.h/R_BHL, R_Hill/R_BHL))

    if wind=="On":
        #eqn 7 from Chen Ren Dai
        r_obd=(vrel2**(1/2) * r_rel)**2 /(ct.G * m) 
        vk=(ct.G * m /r_obd)**(1/2)

        Qco = 2 * alpha * (disk.h/r_obd)**3 * vk**3 /(ct.G * mdot_inflow)

        mdot_obd=[]

        for i in range(0, len(Qco)):
            if Qco[i]>=1:
                mdot_obd.append(mdot_inflow[i])
            elif Qco[i]<1:
                mdot_damp=2 * alpha * (disk.h[i]/r_obd[i])**3 * vk[i]**3 /(ct.G)
                mdot_obd.append(mdot_damp)

        mdot_obd=np.array(mdot_obd)
        Mdot_flux = Mdot * np.abs(1-vstar/vgas)

        mdot_wind = np.minimum(mdot_obd, Mdot_flux)

    elif wind=="Partial":
        mdot_wind=mdot_inflow

    elif wind=="Off":
        mdot_wind=0
    return mdot_wind

def gamma_wind(m, disk, gamma, wind):
    ##from Zhen Pan, Huan Yang (2021), eq. 36
    ##initially developed by Kocsis, Yunes, Loeb (2011)
    h_ratio = disk.h / disk.R
    delta_v_phi = (3. - gamma) / 2. * h_ratio * disk.cs #eq.39a

    mdot_wind = mdot_damped(m, disk, gamma, wind)

    dot_J = - disk.R * delta_v_phi * mdot_wind
    return dot_J


def mig_trap(disk, Gamma):
    maskg = (Gamma >= 0)
    indices = np.nonzero(maskg[1:] != maskg[:-1])[0] + 1
    Gammas = np.split(Gamma, indices)
    Rs = np.split(disk.R, indices)

    ignnum = 0
    radius_trap=[]
    for iseg, seg in enumerate(Gammas):
        if seg[0] < 0.:
            if Rs[iseg][0] / disk.Rs > ignnum + 40:
                radius_trap.append(Rs[iseg][0])
                ignnum = Rs[iseg][0] / disk.Rs
    return radius_trap

def anti_trap(disk, Gamma):
    maskg = (Gamma >= 0)
    indices = np.nonzero(maskg[1:] != maskg[:-1])[0] + 1
    Gammas = np.split(Gamma, indices)
    Rs = np.split(disk.R, indices)

    ignnum = 0
    radius_trap=[]
    for iseg, seg in enumerate(Gammas):
        if seg[0] > 0.:
            if Rs[iseg][0] / disk.Rs > ignnum + 40:
                radius_trap.append(Rs[iseg][0])
                ignnum = Rs[iseg][0] / disk.Rs
    return radius_trap

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

def load_file(filename):
    params = {}
    data = {}
    
    with open(filename, 'r') as file:
        for line in file:
            if line=="Parameters:\n": continue
            elif line=="\n": continue
            elif line=="Data:\n": break
            else:
                line_splitted = line.strip().split()
                if len(line_splitted)==3: 
                    params[line_splitted[0]] = parse(line_splitted[2])
                else: 
                    params[line_splitted[0]] = []
                    for i in range(2, len(line_splitted)): 
                        params[line_splitted[0]].append(parse(line_splitted[i]))

        headers = file.readline().strip().split()
        data = {header: [] for header in headers}

        for line in file:
            values = line.strip().split()
            for i, header in enumerate(headers):
                data[header].append(parse(values[i]))
        
        for header in headers: data[header] = np.array(data[header])
    return params, data

def compute_torque(disk, M, Mbh, TT, wind):
    q = M / Mbh

    ## Kanagawa+2018 prescription
    K= q**2 / disk.alpha  / (disk.h/disk.R)**5
    Sigma = 2. * disk.rho * disk.h
    Sigma_reduced = 1./(1.+0.04*K) * Sigma 
    # NB the position of traps is now m dependent!!
    ##
    
    Gamma_0 = gamma_0(q, disk.h / disk.R, Sigma, disk.R, disk.Omega)
    Gamma_0_reduced = gamma_0(q, disk.h / disk.R, Sigma_reduced, disk.R, disk.Omega)
    Gamma_GW = gamma_GW(disk.R, M, Mbh)
    #Gamma_wind = gamma_wind(M, disk, 5/3)

    dSig_reduced = dSigmadR_reduced(disk, Sigma_reduced)
    dSig = dSigmadR(disk)
    dT = dTdR(disk)

    cI_p10 = CI_p10(disk, dSig_reduced, dT)
    Gamma_I_p10 = cI_p10*Gamma_0_reduced

    cI_jm17 = CI_jm17(dSig_reduced, dT, 5/3, disk)
    Gamma_I_jm17 = cI_jm17*Gamma_0_reduced

    cL = CL(dSig, dT, 5/3, disk)
    Gamma_L = cL*Gamma_0

    exp_decay = np.exp(-K/20.)

    if wind=='On' or "Partial":
        Gamma_wind = gamma_wind(M, disk, 5/3, wind)

    else:
        Gamma_wind = 0

    if TT=="P10": 
        cI_p10 = CI_p10(disk, dSig, dT)
        Gamma_I_p10 = cI_p10*Gamma_0
        return (Gamma_I_p10* exp_decay) + Gamma_GW + Gamma_wind
    if TT=="B16": 
        cI_b16 = CI_b16(disk, dSig, dT)
        Gamma_I_b16 = cI_b16*Gamma_0
        return (Gamma_I_b16 * exp_decay) + Gamma_GW + Gamma_wind
    elif TT=="G23": 
        Gamma_therm = gamma_thermal(5/3, disk, q)*Gamma_0_reduced
        return (Gamma_therm + Gamma_I_jm17)*exp_decay + Gamma_L + Gamma_wind + Gamma_GW

def compute_torque_function(args, disk, M, Mbh):
    Gamma_tot = compute_torque(disk, M, Mbh, args.TT, args.wind)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')

def compute_torque_pure_TypeI(args, disk, M, Mbh):
    q = M / Mbh
    
    Gamma_0 = gamma_0(q, disk.h / disk.R, 2 * disk.rho * disk.h, disk.R, disk.Omega)
    ##Gamma_GW = gamma_GW(disk.R, M, Mbh)

    dSig = dSigmadR(disk)
    dT = dTdR(disk)
    cI_p10 = CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = CI_jm17(dSig, dT, 5/3, disk)
    cL = CL(dSig, dT, 5/3, disk)
    Gamma_I_jm17 = (cL + cI_jm17)*Gamma_0

    if args.TT=="B16": 
        return Gamma_I_p10 ##+ Gamma_GW
    elif args.TT=="G23": 
        gamma = 5/3
        Gamma_therm = gamma_thermal(gamma, disk, q)*Gamma_0
        return Gamma_therm + Gamma_I_jm17  ##+ Gamma_GW

def compute_torque_function_pure_TypeI(args, disk, M, Mbh):
    Gamma_tot = compute_torque_pure_TypeI(args, disk, M, Mbh)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')
    
def rdot(t, y, M, Gamma, M_SMBH, traps):
    return (2*Gamma(y)) / M * np.sqrt(y / (ct.G*M_SMBH))

def rdot_typeII(t, y, disk):
    # Γ = L/t_visc  ==>  ṙ = 2Γ/m √(r/GM) = -2ν/r
    # always negative ==> inward migration
    r = y[0]
    h = np.interp(r, disk.R, disk.h)
    cs = np.interp(r, disk.R, disk.cs)
    nu = disk.alpha * cs * h 
    return -2.0 * nu / r

def rdot_typeII_Kanagawa2018(args, t, y, M, disk, M_SMBH):
    # just like rdot for Type I, but with reduced surface density
    r = float(y[0])

    q=M/M_SMBH
    K= q**2 / disk.alpha  / (disk.h/disk.R)**5
    Sigma = 2. * disk.rho * disk.h
    Sigma_reduced = 1./(1.+0.04*K) * Sigma
    # NB the position of traps is now m dependent!!
    # should also print K or gap depth (0.04*K/(1+0.04*K)) in outputs
    # is it true that you only pair up at low K??
    
    ### Computing torque
    Gamma_0 = gamma_0(q, disk.h / disk.R, Sigma_reduced, disk.R, disk.Omega)
    Gamma_0_background = gamma_0(q, disk.h / disk.R, Sigma, disk.R, disk.Omega)

    dSig = dSigmadR(disk)
    dT = dTdR(disk)

    cI_jm17 = CI_jm17(dSig, dT, 5/3, disk)
    gamma = 5/3
    cL = CL(dSig, dT, gamma, disk)
    Gamma_I_jm17 = cL*Gamma_0_background + cI_jm17*Gamma_0

    if args.TT=="P10": 
        cI_p10 = CI_p10(disk, dSig, dT)
        Gamma_I_p10 = cI_p10*Gamma_0
        Gamma = Gamma_I_p10 ##+ Gamma_GW
    if args.TT=="B16": 
        cI_b16 = CI_b16(disk, dSig, dT)
        Gamma_I_b16 = cI_b16*Gamma_0
        Gamma = Gamma_I_b16 ##+ Gamma_GW
    elif args.TT=="G23": 
        gamma = 5/3
        Gamma_therm = gamma_thermal(gamma, disk, q)*Gamma_0
        Gamma = Gamma_therm + Gamma_I_jm17 #+ Gamma_I_p10 ##+ Gamma_GW
    ###

    Gamma_of_r = interp1d(disk.R, Gamma, kind="linear", fill_value="extrapolate")
    Gamma_r = float(Gamma_of_r(r))

    return (2.0 * Gamma_r / M) * np.sqrt(r / (ct.G * M_SMBH))


def trap_dist(t, y, M, Gamma, M_SMBH, traps):
    return np.prod([y-trap for trap in traps])
trap_dist.terminal = True # stops integration

def trap_dist_plot(t, y, M, Gamma, M_SMBH, traps):
    return [trap_dist(t, r, M, Gamma, M_SMBH, traps) for r in y]

def first_root(func, domain):
    for i in range(len(domain)-1):
        if func(domain[i])*func(domain[i+1])<0:
            return fsolve(func, (domain[i]+domain[i+1])/2)[0]
    
def pos_after_kick(r_init, mass_prim_vk, M_SMBH):
    vks = mass_prim_vk[:, 2]
    a = np.random.randint(0,len(vks))
    vk = vks[a]*1e3

    mass_enclosed = 0 # we neglect mass_enclose/M_SMBH < 1e-4
    
    vesc = np.sqrt(2 * ct.G * (M_SMBH + mass_enclosed*ct.MSun) / r_init)
    v_kepler = vesc / np.sqrt(2)
    
    rnd_theta = np.arccos(1 - 2*np.random.rand())
    rnd_phi = 2 * np.pi * np.random.rand()
    v_total = np.sqrt((v_kepler + np.cos(rnd_phi)*np.sin(rnd_theta)*vk)**2 + (np.sin(rnd_phi)*np.sin(rnd_theta)*vk)**2 + (np.cos(rnd_theta)*vk)**2)
    
    r_new = r_init/2 /(1 - (v_total/vesc)**2)
    
    return r_new

def type_II_event(disk, M, Mbh):
    #Checks whether a BH of mass M opens a gap in the AGN disk
    q_array = M / Mbh
    h_array = disk.h / disk.R
    R_array = disk.R

    from scipy.interpolate import interp1d
    h_of_r = interp1d(R_array, h_array, bounds_error=False, fill_value="extrapolate")
    
    # 1) The disk is not too viscous 
    # "Classic" Type II
    def viscous_event(t, y, *fargs):
        r = y[0]
        h = h_of_r(r)
        q = q_array
        return q - np.sqrt(disk.alpha / 0.09) * h**5 #K>11
    viscous_event.terminal = False # doesn't stop integration
    viscous_event.direction = 1  # Trigger when q crosses threshold from below

    
    # 2) The disk is thin enough 
    # from Bryden+99 and citations therein
    def thin_event(t, y, *fargs):
        r = y[0]
        h = h_of_r(r)
        q = q_array
        return h - (q / 3.0)**(1/3) #*r**2
    thin_event.terminal = False # doesn't stop integration
    thin_event.direction = -1  # Trigger when h crosses threshold from above

    # "and" condition
    def gap_open_event(t, y, *fargs):
        v = viscous_event(t, y, *fargs)
        return -v 
    gap_open_event.terminal = False # doesn't stop integration
    gap_open_event.direction = -1  # Trigger when h crosses threshold from above
    
    return gap_open_event
    #return viscous_event

def dummy_event(*args, **kwargs):
    return 1.0  # Always positive — no zero crossing
dummy_event.terminal = False
dummy_event.direction = 0
################################################################################################