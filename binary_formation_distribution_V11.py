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
plotting = False
printing = True
type_II_computation = "modified_type_I" 
##can be "conservative" or "conservative+extra_h_condition" or "modified_type_I"
initialize_radius = 'capture_probability'
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

def capture_processing(disk, M, Rmin, Rmax):
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
    ##cI = -0.85 + 0.9*dTdR + dSigmadR #Paardekooper et al. 2010 (iso)
    cI_iso = gamma_iso(dSigmadR, dTdR)
    cI_ad = gamma_ad(dSigmadR, dTdR)

    cI = (cI_ad * Theta(obj)**2. + cI_iso) / (Theta(obj) + 1.)**2.

    return cI

def CI_jm17(dSigmadR, dTdR, gamma, obj):
    cI = (0.46 + 0.96*dSigmadR - 1.8*dTdR)/gamma
    return cI


# modified by Paola Vaccaro
def CL(dSigmadR, dTdR, gamma, obj):
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
    return g_thermal*decay *(obj.R/obj.h) ##new 1/h term wtr V7
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

def mdot_damped(m, disk, gamma, r):
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
    return mdot_wind

def gamma_wind(m, disk, gamma):
    ##from Zhen Pan, Huan Yang (2021), eq. 36
    ##initially developed by Kocsis, Yunes, Loeb (2011)
    h_ratio = disk.h / disk.R
    delta_v_phi = (3. - gamma) / 2. * h_ratio * disk.cs #eq.39a

    mdot_wind = mdot_damped(m, disk, gamma, disk.R)

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

def compute_torque(disk, M, Mbh, TT):
    q = M / Mbh

    ## Kanagawa+2018 prescription
    K= q**2 / disk.alpha  / (disk.h/disk.R)**5
    Sigma_reduced = 1./(1.+0.04*K) * (2. * disk.rho * disk.h)
    # NB the position of traps is now m dependent!!
    # is it true that you only pair up at low K??
    ##
    
    #Gamma_0 = gamma_0(q, disk.h / disk.R, 2 * disk.rho * disk.h, disk.R, disk.Omega)
    Gamma_0 = gamma_0(q, disk.h / disk.R, Sigma_reduced, disk.R, disk.Omega)
    Gamma_GW = gamma_GW(disk.R, M, Mbh)
    Gamma_wind = gamma_wind(M, disk, 5/3, Mbh)

    dSig = dSigmadR_reduced(disk, Sigma_reduced)
    dT = dTdR(disk)
    cI_p10 = CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = CI_jm17(dSig, dT, 5/3, disk)
    cL = CL(dSig, dT, 5/3, disk)
    Gamma_I_jm17 = (cL + cI_jm17)*Gamma_0

    if TT=="B16": 
        return Gamma_I_p10 ##+ Gamma_GW
    elif TT=="G23": 
        gamma = 5/3
        Gamma_therm = gamma_thermal(gamma, disk, q)*Gamma_0
        return Gamma_therm + Gamma_I_jm17 + Gamma_wind + Gamma_GW

def compute_torque_function(args, disk, M, Mbh):
    Gamma_tot = compute_torque(args, disk, M, Mbh)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')

def compute_torque_pure_TypeI(args, disk, M, Mbh):
    q = M / Mbh
    
    Gamma_0 = gamma_0(q, disk.h / disk.R, 2 * disk.rho * disk.h, disk.R, disk.Omega)
    ##Gamma_GW = gamma_GW(disk.R, M, Mbh)

    dSig = dSigmadR_reduced(disk, 2 * disk.rho * disk.h)
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
        return Gamma_therm + Gamma_I_jm17 #+ Gamma_I_p10 ##+ Gamma_GW

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

def rdot_typeII_Kanagawa2018(t, y, M, disk, M_SMBH):
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
    cI_p10 = CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = CI_jm17(dSig, dT, 5/3, disk)
    gamma = 5/3
    cL = CL(dSig, dT, gamma, disk)
    Gamma_I_jm17 = cL*Gamma_0_background + cI_jm17*Gamma_0

    if args.TT=="B16": 
        Gamma = Gamma_I_p10 ##+ Gamma_GW
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
        if type_II_computation == "conservative+extra_h_condition":
            v   = viscous_event(t, y, *fargs)
            th  = thin_event(t, y, *fargs)
            return max(-v, th)           # ≤ 0 only when BOTH are satisfied
        elif (type_II_computation == "conservative") or (type_II_computation == "modified_type_I"):
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


################################################################################################
### Simulation routine #########################################################################
################################################################################################
def iteration(Rmin, Rmax, args, mass_sec, mass_prim_vk, disk, Mbh, r_pu_1g):
    # initialize random number generator for the radii and the masses
    seed = (os.getpid() + int(time.time() * 1e6)) % 2**32
    np.random.seed(seed)
    
    # primary and secondary mass
    a = np.random.randint(0,len(mass_sec),2)
    (m1,m2) = mass_sec[a]*ct.MSun
    if args.gen=='Ng':
        mass_prim = mass_prim_vk[:, 0]
        a = np.random.randint(0,len(mass_prim))
        m1 = mass_prim[a]*ct.MSun
        Ng = mass_prim_vk[a, 2]
    elif args.gen=='1g':
        Ng = 1
        
    M = np.array([m1, m2]) if m1>m2 else np.array([m2, m1])
    #type II migration flag
    gap = np.full(M.shape, 'type_I', dtype='<U7')
    K = np.zeros(M.shape)

    # time and initial radius
    t0 = 0
    if initialize_radius == 'log_uniform':
        r0 = 10**(np.log10(Rmin) + (np.log10(Rmax) - np.log10(Rmin)) * np.random.rand(2))
    elif initialize_radius == 'capture_probability':
        r0 = capture_processing(disk, M, Rmin, Rmax) 
        
    if args.gen=='Ng':
        b = np.random.randint(0, len(r_pu_1g))
        r1 = r_pu_1g[b]
        r0[0] = pos_after_kick(r1, mass_prim_vk, Mbh)
        if not np.isfinite(r0[0]):
            r0[0]=Rmax+1e10
    T = args.T * ct.yr

    # compute torque & trap locations for both BHs
    Gammas = [compute_torque_function(args, disk, M[i], Mbh) for i in range(2)]
    traps = [mig_trap(disk, Gamma(disk.R)) for Gamma in Gammas]
    
    Gammas_pure_TypeI = [compute_torque_function_pure_TypeI(args, disk, M[i], Mbh) for i in range(2)]
    traps_pure_TypeI = [mig_trap(disk, Gamma(disk.R)) for Gamma in Gammas_pure_TypeI]
    
     
    # type II migration in case of gap opening
    gap_event_1 = type_II_event(disk, M[0], Mbh) 
    gap_event_2 = type_II_event(disk, M[1], Mbh)

    # simulate trajectories until t>T or until r=r_trap
    solution1 = solve_ivp(fun=rdot, 
                          t_span=[t0, T], 
                          y0=[r0[0]],
                          method='RK23', 
                          events=(trap_dist, gap_event_1), 
                          first_step=1e3*ct.yr,
                          rtol=1e-3,
                          atol=1e-9,
                          args=(M[0], Gammas[0], Mbh, traps[0]))
    solution2 = solve_ivp(fun=rdot, 
                          t_span=[t0, T], 
                          y0=[r0[1]],
                          method='RK23', 
                          events=(trap_dist, gap_event_2), 
                          first_step=1e3*ct.yr,
                          rtol=1e-3,
                          atol=1e-9,
                          args=(M[1], Gammas[1], Mbh, traps[1]))
    


    ## check whether "events" happened (reaching Type I migration trap or doing Type II migration)
    t1 = solution1.t
    t2 = solution2.t
    r1 = solution1.y[0]
    r2 = solution2.y[0]

    # if you open a gap
    # just print type II flag
    if len(solution1.t_events[1]) >0:
        gap[0]='type_II'
        """
        t_gap = solution1.t_events[1][0]
        r_gap = solution1.y_events[1][0][0]
        # Stage 2 : Type II migration
        if (type_II_computation == "conservative") or (type_II_computation == "conservative+extra_h_condition"):
            solution1_TypeII = solve_ivp(rdot_typeII, [t_gap, T], [r_gap],
                                     args=(disk,),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        elif type_II_computation == "modified_type_I":
            solution1_TypeII = solve_ivp(rdot_typeII_Kanagawa2018, [t_gap, T], [r_gap],
                                     args=(M[0], disk, Mbh,),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        # Stitch the two segments 
        t1 = np.concatenate((t1, solution1_TypeII.t[1:]))     # skip duplicate t_gap
        r1 = np.concatenate((r1, solution1_TypeII.y[0][1:]))
        """
    else:
        # if BH reached migration trap
        # extend trajectory until t=T
        if len(solution1.t_events[0])>0:
            t1 = np.append(t1, T)
            r1 = np.append(r1, r1[-1])
    if len(solution2.t_events[1]) >0:
        gap[1]='type_II'
        """
        t_gap = solution2.t_events[1][0]
        r_gap = solution2.y_events[1][0][0]
        # Stage 2 : Type II migration
        if (type_II_computation == "conservative") or (type_II_computation == "conservative+extra_h_condition"):
            solution2_TypeII = solve_ivp(rdot_typeII, [t_gap, T], [r_gap],
                                         args=(disk,),
                                         first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        elif type_II_computation == "modified_type_I":
            solution2_TypeII = solve_ivp(rdot_typeII_Kanagawa2018, [t_gap, T], [r_gap],
                                     args=(M[1], disk, Mbh,),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        # Stitch the two segments 
        t2 = np.concatenate((t2, solution2_TypeII.t[1:]))     # skip duplicate t_gap
        r2 = np.concatenate((r2, solution2_TypeII.y[0][1:]))
        """
    else:
        # if BH reached migration trap
        # extend trajectory until t=T
        if len(solution2.t_events[0])>0:
            t2 = np.append(t2, T)
            r2 = np.append(r2, r2[-1])
    
    # interpolate trajectory
    r1_itp = interp1d(t1, r1, kind='linear', fill_value='extrapolate')
    r2_itp = interp1d(t2, r2, kind='linear', fill_value='extrapolate')

    # two conditions for pair-up
    # 1) crossing trajectories
    def cross(t):
        return r1_itp(t)-r2_itp(t)

    # 2) distance between both being smaller than the Hill radius of the primary
    def Hill_range(t):
        rH = max(r1_itp(t)*np.cbrt(M[0]/(3*(Mbh+M[0]))), r2_itp(t)*np.cbrt(M[1]/(3*(Mbh+M[1]))))
        return abs(r1_itp(t)-r2_itp(t)) - rH
    
    # Find pair up information
    r_pu, t_pu, paired, r_new = 0, 0, 0, 0
    
    try:
        t = np.logspace(0, np.log10(T), 1000)
        
        # first check wether trajectories crossed
        pair = first_root(cross, t)
        if pair!=None: 
            paired = 1
            t_pu = pair
            r_pu = (M[0]*r1_itp(t_pu) + M[1]*r2_itp(t_pu)) / (np.sum(M))
        else: 
            # if not, check for Hill condition
            pair = first_root(Hill_range, t)
            if pair!=None: 
                paired = 1
                t_pu = pair
                r_pu = (M[0]*r1_itp(t_pu) + M[1]*r2_itp(t_pu)) / (np.sum(M))
    except RuntimeWarning: 
        # if no root is found assume pair up did not occur
        pass
        
    if args.plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(8, 9))
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(1, T/ct.yr)
        ax1.set_ylim(Rmin, Rmax)
        ax1.plot(t1/ct.yr, r1, c='C0', ls='solid')
        ax1.plot(t2/ct.yr, r2, c='C1', ls='solid')
        ax1.vlines(t_pu/ct.yr, Rmin, Rmax, color='gray', ls='dashed')
        ax1.hlines(r_pu, 1, T/ct.yr, color='gray', ls='dashed')
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(1, T/ct.yr)
        ax2.set_ylim(1, Rmax**max(len(traps[0]), len(traps[1])))
        ax2.plot(solution1.t/ct.yr, np.abs(trap_dist_plot(solution1.t, solution1.y.T, M[0], Gammas[0], Mbh, traps[0])))
        ax2.plot(solution2.t/ct.yr, np.abs(trap_dist_plot(solution2.t, solution2.y.T, M[1], Gammas[1], Mbh, traps[1])))
        ax2.text(1, 1e30, f"{len(solution1.t)} and {len(solution2.t)}")
        
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.plot(disk.R, np.abs(Gammas[0](disk.R)))
        ax3.plot(disk.R, np.abs(Gammas[1](disk.R)))
        
        plt.tight_layout()
        plt.show()
    

    #contrast parameter
    h_of_r = interp1d(disk.R, disk.h/disk.R, bounds_error=False, fill_value="extrapolate")
    h = h_of_r(r_pu)
    K = (M / Mbh)**2 /disk.alpha /h**5

    return f"{r0[0]:.3e} {r0[1]:.3e} {M[0]:.3e} {M[1]:.3e} {r_pu:.3e} {t_pu:.3e} {paired} {gap[0]} {gap[1]} {Ng} {K[0]} {K[1]}\n"
################################################################################################

################################################################################################
### Plot |Gamma(R)| for a range of BH masses for a given SMBH###################################
################################################################################################
def plot_torque_vs_radius(args, disk, Mbh, mass_range, Mmean):
    Rsch = 2 * ct.G * Mbh / ct.c**2
    Rs = disk.R / Rsch
    Gammas = []

    for M in mass_range:
        Gamma = compute_torque(args, disk, M, Mbh)
        Gammas.append(Gamma)

    Gammas = np.array(Gammas)
    #mean_Gamma = np.mean(Gammas, axis=0)
    mean_Gamma = compute_torque(args, disk, Mmean, Mbh)
    traps = mig_trap(disk, mean_Gamma) 
    traps=np.array(traps)/Rsch
    min_Gamma = np.min(Gammas, axis=0)
    max_Gamma = np.max(Gammas, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(Rs, np.abs(min_Gamma), np.abs(max_Gamma), alpha=0.3, label='Migrator mass range')
    for trap in traps:
        ax.axvline(trap, color='gray', linestyle=':', linewidth=1, label='Migration trap' if trap == traps[0] else "")

    # Mean line: solid for Gamma > 0, dashed for Gamma < 0
    # Split into continuous regions of positive or negative torque
    sign_changes = np.where(np.diff(np.sign(mean_Gamma)) != 0)[0]
    split_indices = np.concatenate(([0], sign_changes + 1, [len(Rs)]))

    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        R_seg = Rs[start:end]
        G_seg = mean_Gamma[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='|Gamma| >0' if i == 1 else "")
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='|Gamma| <0' if i == 0 else "")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'R [R$_{\rm S}$]')
    ax.set_ylabel(r'$|\Gamma|$ [cgs]')
    ax.set_ylim([1e29,1e44])
    ax.set_title(f'Type I migration Torques (log M_SMBH={np.log10(Mbh/ct.MSun):.1f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'gammas_frames/Mbh_{np.log10(Mbh/ct.MSun):.1f}_alpha_{args.a}_{args.DT}_{args.TT}_{args.gen}.png', format='png', dpi=300)
    return
################################################################################################



################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-M_SMBH', type=float, default=7)  # real number
    parser.add_argument('-a', type=float, default=0.01)    # real number
    parser.add_argument('-T', type=float, default=2e6)     # real number [yr]
    parser.add_argument('-N', type=int, default=10000) # integer number
    parser.add_argument('-plot', action='store_true')      # truth value
    parser.add_argument('-date', action='store_true')      # truth value
    
    args = parser.parse_args()
    return args
################################################################################################



################################################################################################
### Loading information on mass distribution ###################################################
################################################################################################
if __name__ == '__main__':
    args=main()
    mass_sec=np.genfromtxt("src/BHs_single_Zsun_rapid_nospin.dat",usecols=(0),skip_header=3,unpack=True)
    mass_prim_vk = np.genfromtxt('src/Ng_catalog.txt', skip_header=1)
################################################################################################



################################################################################################
### Initialization of the Disk #################################################################
################################################################################################
    Mbh = 10**args.M_SMBH * ct.MSun  # M_SMBH
    alpha = args.a                   # viscosity parameter

    if args.DT  == "SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha)
        Rmin = disk.Rmin
        Rmax = disk.Rmax
    elif args.DT  == "TQM":
        disk = pagn.ThompsonAGN(Mbh=Mbh, Mdot_out=None)
        Rmin = disk.Rin
        Rmax = disk.Rout
        print(disk.Mdot_out)
    disk.solve_disk()
################################################################################################



################################################################################################
### Computation of migration trap positions ####################################################
################################################################################################
    Torque = compute_torque(args, disk, np.mean(mass_sec) * ct.MSun, Mbh)
    ##print(np.mean(mass_sec))
    traps = mig_trap(disk, Torque)

    Torque = compute_torque_pure_TypeI(args, disk, np.mean(mass_sec) * ct.MSun, Mbh)
    traps_pure_TypeI = mig_trap(disk, Torque)
################################################################################################

################################################################################################
### Plot migration torques magnitudes ##########################################################
################################################################################################
    if plotting == True:
        plot_torque_vs_radius(args, disk, Mbh, np.linspace(5, 50, 10) * ct.MSun, np.mean(mass_sec) * ct.MSun)
        ##print("np.mean(mass_sec) =", np.mean(mass_sec))
################################################################################################


################################################################################################
### Output file ################################################################################
################################################################################################
    if printing == True:
        traps_str = ""
        for trap in traps: traps_str += f"{trap:.1e} "
        traps_str_pure_TypeI = ""
        for trap in traps_pure_TypeI: traps_str_pure_TypeI += f"{trap:.1e} "

        date_time = start.strftime("%y%m%d_%H%M%S")

        dir_name = f"outputs_V9/{args.DT}/alpha_{args.a}/Mbh_{args.M_SMBH:.1f}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name+f"/pairup_radius_{args.TT}_{args.gen}.txt"
        file_name_1g = dir_name+f"/pairup_radius_{args.TT}_1g.txt"
        if args.gen=='Ng' and  not os.path.exists(file_name_1g):
            print()
            print('There is no 1g source for this Ng simulation. Run the same simulation for 1g first!')
            quit()
        file = open(file_name, "w")

        # print all parameters in the file
        file.write(f"Parameters:\n")
        file.write(f"version     = V9\n")
        file.write(f"date_time   = {date_time}\n")
        file.write(f"comp_time   = {0}\n")
        file.write(f"disk_type   = {args.DT}\n")
        file.write(f"torque_type = {args.TT}\n")
        file.write(f"log(M_SMBH) = {args.M_SMBH:.1f}\n")
        file.write(f"alpha       = {args.a:.3f}\n")
        file.write(f"T           = {args.T:.1e}\n")
        file.write(f"gen         = {args.gen}\n")
        file.write(f"N           = {args.N}\n")
        file.write(f"R_min       = {Rmin:.1e}\n")
        file.write(f"R_max       = {Rmax:.1e}\n")
        file.write(f"trap_rad    = {traps_str}\n")
        file.write(f"trap_rad_TypeI    = {traps_str_pure_TypeI}\n")
        file.write(f"\n")
        file.write(f"Data:\n")
        file.write(f"r_1       r_2       m_1       m_2       r_pu      t_pu      paired      migration_1      migration_2      Ng_1      K_1      K_2\n")
################################################################################################


################################################################################################
### Loading of 1g r_pu data ####################################################################
################################################################################################
        if args.gen=='1g':
            r_pu_1g = np.array([])
        if args.gen=='Ng':
            params, data = load_file(file_name_1g)
            r_pu_1g = data['r_pu'][data['paired']==1]
################################################################################################

################################################################################################
### Determination of binary formation radii ####################################################
################################################################################################
        print()
        N_batches = 1000
        N_iter = int(args.N)
        chunk_size = int(N_iter/N_batches)

        # run parallel simulations and print results in file
        #with multiprocessing.Pool(1) as pool:
        with multiprocessing.Pool(os.cpu_count()) as pool:
            with tqdm(total=N_iter) as pbar:
                for i in range(N_batches):
                    input_data = [(Rmin, Rmax, args, mass_sec, mass_prim_vk, disk, Mbh, r_pu_1g) for _ in range(chunk_size)]
                    results = pool.starmap(iteration, input_data)
                    if len(results) != chunk_size:
                        print(f"[Warning] Batch {i} returned {len(results)} runs instead of {chunk_size}")
                    file.writelines(results)
                    pbar.update(chunk_size)
################################################################################################


################################################################################################
### Write duration in file and close file ######################################################
################################################################################################
        file = open(file_name, "r")
        lines = file.readlines()
        file = open(file_name, "w")
        for l in lines:
            if 'comp_time' in l: 
                comp_time = datetime.now() - start
                file.writelines(f"comp_time   = {comp_time}\n")
            else: file.writelines(l)
        file.close()
################################################################################################