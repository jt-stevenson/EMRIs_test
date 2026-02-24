import numpy as np

import matplotlib.pyplot as plt
import multiprocessing
import binary_formation_distribution_V8 as myscript
from scipy.interpolate import interp1d

import pagn
import powerlaw
import imf
from os import makedirs
import pandas as pd

from scipy.interpolate import UnivariateSpline

#constants from pagn.constants,had some issues loading the constants for some reason so just copied them across locally
c = 2.99792458e8  # m/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
MSun = 1.98847e30  # kg
yr = 365.25*24*60*60  # s
pc = 3.086e16  # m
RSun_to_au = 4.6491303438174012e-03  # R_sol/AU
SI_to_gcm3 = 1e-3
SI_to_gcm2 = 1e-1
SI_to_cms = 1e2

K_to_eV=8.6173e-5

def R_isco_function(MBH, spin):
    #function to calculate innermost stable circular orbit for a BH of given mass and spin
    R_G=G*MBH*(1/(c*c))
    Z_1= 1 + ((1-(spin*spin))**(1/3)) * ((1+spin)**(1/3) + (1-spin)**(1/3))
    Z_2=(3*spin*spin + Z_1*Z_1)**(1/2)
    R_isco=R_G*(3+Z_2 - ((3-Z_1)*(3+Z_1+2*Z_2))**(1/2))
    # print(f'R_isco={R_isco}')
    return R_isco

# Eqns 99 from Abramowicz and Fragile for NT Disk Profile

def y_fns(MBH, spin):
    M=MBH * G /(c*c)
    r_isco=R_isco_function(MBH, spin)

    y0=np.sqrt(r_isco/M)

    y1=2*np.cos((np.arccos(spin)- np.pi)*(1/3))
    y2=2*np.cos((np.arccos(spin)+ np.pi)*(1/3))
    y3=-2*np.cos((np.arccos(spin))*(1/3))
    return y0, y1, y2, y3

def A_fn(y, spin):
    A = 1 + (spin*spin * y**(-4)) + (2 * spin*spin * y**(-6))
    return A

def B_fn(y, spin):
    B = 1 + (spin * y**(-3))
    return B

def C_fn(y, spin):
    C = 1 - (3 * y**(-2)) + (2 * spin * y**(-3))
    return C

def D_fn(y, spin):
    D = 1 - (2 * y**(-2)) + (spin*spin * y**(-4))
    return D

def E_fn(y, spin):
    E = 1 + (4 * spin*spin * y**(-4)) - (4 * spin*spin * y**(-6)) + (3 * spin*spin*spin*spin * y**(-8))
    return E

def Q_fn(y, MBH, spin):
    y0, y1, y2, y3= y_fns(MBH, spin)
    num = 1 + (spin * y**(-3))
    denom = y * (1 - 3*y**(-2) +2*spin*y**(-3))**(1/2)
    Q0 = num/denom
    
    term1= 3 * ((y1-spin)**2) / (y1 * (y1-y2)* (y1-y3)) * np.log((y-y1)/(y0-y1))
    term2= 3 * ((y2-spin)**2) / (y2 * (y2-y1)* (y2-y3)) * np.log((y-y2)/(y0-y2))
    term3= 3 * ((y3-spin)**2) / (y3 * (y3-y1)* (y3-y2)) * np.log((y-y3)/(y0-y3))

    Q=Q0*(y - y0 - (3/2)*spin*np.log(y/y0) - term1 - term2 - term3)
    return Q

#Surface Densities
def Sigma_NT(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    sigma = 5 * (1/alpha) * (1/mdot) * rstar**(3/2) * (1/(A*A)) * B*B*B * C**(1/2) * E * (1/Q)
    return sigma

def Sigma_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    sigma = (9e4) * alpha**(-4/5) * m**(1/5) * mdot**(3/5) * rstar**(-3/5) * B**(-4/5) * C**(1/2) * D**(-4/5) * Q**(3/5)
    return sigma

def Sigma_NT_Outer(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    sigma = (4e5) * alpha**(-4/5) * m**(1/5) * mdot**(7/10) * rstar**(-3/4) * A**(1/20) * B**(-4/5) * C**(1/2) * D**(-17/20) * E**(-1/20) * Q**(7/10)
    return(sigma)

#Thicknesses
def H_NT(r, MBH, spin, mdot):
    #accurate to A+F
    M=MBH * G /(c*c)
    y=np.sqrt(r/M)
    
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    H = 1e5 * mdot * A**2 * B**(-3) * C**(1/2) * D**(-1) * E**(-1) * Q #in cms
    return H

def H_NT_2(r, MBH, spin, mdot):
    #added factor of m like in Grishin et al 2025, seems to fix a scaling issue
    M=MBH * G /(c*c)
    m=MBH/MSun
    y=np.sqrt(r/M)
    
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    H = 1e5 * mdot * m * A**2 * B**(-3) * C**(1/2) * D**(-1) * E**(-1) * Q #in cms
    return H

def H_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)
    
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    H = 1e3 * alpha**(-1/10) * m**(9/10) * mdot**(1/5) * rstar**(21/20) * A * B**(-6/5) * C**(1/2) * D**(-3/5) * E**(-1/2) * Q**(1/5) #in cms
    return H

def H_NT_Outer(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)
    
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    H = 4e2 * alpha**(-1/10) * m**(18/20) * mdot**(3/20) * rstar**(9/8) * A**(19/20) * B**(-11/10) * C**(1/2) * D**(-23/40) * E**(-19/40) * Q**(3/20) #in cms
    return H

#Temperatures
def T_NT(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    E=E_fn(y, spin)

    T= (5e7) * alpha**(-1/4) * m**(-1/4) * rstar**(-3/8) * A**(-1/2) * B**(1/2) * E**(1/4) #in Kelvin
    return T

def T_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    B=B_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    T = (7e8) * alpha**(-1/5) * m**(-1/5) * mdot**(2/5) * rstar**(-9/10) * B**(-2/5) * D**(-1/5) * Q**(2/5)
    return(T)

def T_NT_Outer(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    T = (2e8) * alpha**(-1/5) * m**(-1/5) * mdot**(3/10) * rstar**(-3/4) * A**(-1/10) * B**(-1/5) * D**(-1/5) * D**(-3/10) * Q**(3/5)
    return(T)

#Midplane Densities
def rho_0_NT(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    rho_0 = 2e-5 * alpha**(-1) * m**(-1) * mdot**(-2) * rstar**(3/2) * A**(-4) * B**6 * D * E*E * Q**(-2)
    return(rho_0)

def rho_0_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    rho_0 = 4e1 * alpha**(-7/10) * m**(-7/10) * mdot**(2/5) * rstar**(-33/20) * A**(-1) * B**(3/5) * D**(-1/5) * E**(1/2) * Q**(2/5)
    return(rho_0)

def rho_0_NT_Outer(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    rho_0 = 4e2 * alpha**(-7/10) * m**(-7/10) * mdot**(11/20) * rstar**(-15/8) * A**(-17/20) * B**(3/10) * D**(-11/40) * E**(17/40) * Q**(11/20)
    return(rho_0)

#Irradiance
def F_NT(r, MBH, spin, mdot):
    #Same for inner, mid and outer regions
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    B=B_fn(y, spin)
    C=C_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 7e26 * m**(-1) * mdot * rstar**(-3) * B**(-1) * C**(-1/2) * Q #in erg cm^-2 s^-1
    return output

# Radiation Pressure
def Beta_NT(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 4e-6 * alpha**(-1/4) * m**(-1/4) * mdot**(-2) * rstar**(21/8) * A**(-5/2) * B**(9/2) * D * E**(5/4) * Q**(-2) #unitless
    beta = output/(1+output)
    return beta

def Beta_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 7e-3 * alpha**(-1/10) * m**(-1/10) * mdot**(-4/5) * rstar**(21/20) * A**(-1) * B**(9/5) * D**(2/5) * E**(1/2) * Q**(-4/5) #unitless
    beta = output/(1+output)
    return beta

def Beta_NT_Outer(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 3 * alpha**(-1/10) * m**(-1/10) * mdot**(-7/10) * rstar**(3/8) * A**(-11/20) * B**(9/10) * D**(7/40) * E**(11/40) * Q**(-7/20) #unitless
    beta = output/(1+output)
    return beta

# Optical Depth Equations
def Tau_NT(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 1e-4 * alpha**(-17/16) * m**(-1/16) * mdot**(-2) * rstar**(93/32) * A**(-25/8) * B**(41/8) * C**(1/2) * D**(1/2) * E**(25/16) * Q**(-2) #unitless
    tau_ff_tau_es=output**2
    return tau_ff_tau_es

def Tau_NT_Middle(r, MBH, spin, mdot):
    M=MBH * G /(c*c)
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 2e-6 * mdot**(-1) * rstar**(3/2) * A**(-1) * B**(2) * D**(1/2) * E**(1/2) * Q**(-1) #in units
    tau_ff_div_tau_es=output
    return tau_ff_div_tau_es

def Tau_NT_Outer(r, MBH, spin, mdot):
    M=MBH * G /(c*c)
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = 2e-3 * mdot**(-1/2) * rstar**(3/4) * A**(-1/2) * B**(2/5) * D**(1/4) * E**(1/4) * Q**(-1/2) #in units
    tau_ff_div_tau_es=output
    return tau_ff_div_tau_es

def Template_NT(r, MBH, spin, mdot, alpha):
    # Template NT function to make coding the rest of the equations easier
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    output = alpha**(1) * m**(1) * mdot**(1) * rstar**(1) * A**(1) * B**(1) * C**(1) * D**(1) * E**(1) * Q**(1) #in units
    return output

#Eqns from Krolik 1999 to check where transition between inner and outer SS (and thereby NT) disc equations is
#And if our assumption is valid only using inner equations

def R_Rfn(y, MBH, spin):
    Q=Q_fn(y, MBH, spin)
    C=C_fn(y, spin)
    B=B_fn(y, spin)
    rR=Q * C**(-1/2) *B**(-1)
    return rR

def R_Tfn(y, MBH, spin):
    Q=Q_fn(y, MBH, spin)
    C=C_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    rT= Q * C**(1/2) * B**(-1) * D**(-1)
    return rT

def L_fn(y, spin):
    C=C_fn(y, spin)
    L=y * (1 - 2 * spin * y**(-3) + spin * y**(-4)) * C**(-1/2)
    return L

def Einf_fn(y, spin):
    C=C_fn(y, spin)
    Einf= (1 - 2 * y**(-2) + spin * y**(-3)) * C**(-1/2)
    return Einf

def R_Zfn(y, spin):
    L=L_fn(y, spin)
    Einf=Einf_fn(y, spin)
    rZ=y**(-2) * (L*L - spin*spin*(Einf-1))
    return rZ

def R_tr(y, MBH, spin, eps, le, alpha):
    #Davis and Tchekhovskoy 2021
    #approx location of transition between inner and outer regions for SS disc
    rR=R_Rfn(y, MBH, spin)
    rT=R_Tfn(y, MBH, spin)
    rZ=R_Zfn(y, spin)
    M=MBH * G /(c*c)
    L_term = (0.1 * eps**(-1) * le * 10)**(16/21)
    M_term = (10 * alpha * MBH * (1e8 * MSun)**(-1))**(2/21)
    R_term = rR**(6/7) * rZ**(-10/21) * rT**(-2/21)
    rTR=340 * M * L_term * M_term * R_term
    return rTR

#Eqns from Grishin et al 2025 defining transitions between inner, middle and outer regions

def R_inner_mid(r, MBH, mdot, alpha):
    m=MBH/(1e8*MSun)
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=449.842 * alpha**(2/21) * m**(2/21) * mdotprime**(16/21)
    return(r_tr)

def R_mid_outer(r, mdot):
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=987.891 * mdotprime**(2/3)
    return(r_tr)

def R_outer_AGN(r, MBH, mdot, alpha):
    m=MBH/(1e8*MSun)
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=580.65 * alpha**(28/45) * m**(-52/45) * mdotprime**(-22/45)
    return(r_tr)

#Eqns derived from A+F NT Surface Density Profile Eqns

def r_in_mid(y, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    r_tr=(4.5e4)**(10/21) * (alpha**2 * m**2 * mdot**16 * A**20 * B**-38 * D**-8 * E**-10 * Q**16)**(1/21)
    return(r_tr)

def r_mid_outer(y, MBH, spin, mdot):
    A=A_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)
    r_tr=49**(20/3) * mdot**(2/3) * A**(2/3) * D**(-1/3) * Q**(2/3)
    return(r_tr)

# Eqns to calculate GW frequency and when an EMRI enters the LISA band

def GW_freq_fn(r, MBH, m):
    M=MBH+m
    f= 1/np.pi * (G * M * 1/(r*r*r))**(1/2)
    return f


def LISAband_flag(Rstart, Rmin, MBH, m):
    lisa_flag=0
    lisa_radii=0
    R=np.linspace(Rstart, Rmin, 10000)
    for r in R:
        GWf=GW_freq_fn(r, MBH, m)
        if 1.0>GWf>0.0001 and lisa_flag==0:
            lisa_radii+=r
            lisa_flag+=1
            break
    return lisa_flag, lisa_radii

# defined event to check if SBH has hit the event horizon
def r_s_event(MBH, m):
    Rs=2*G*MBH /(c*c)
    Rs_sbh=2*G*m /(c*c)
    def crossing_event(t,y, *fargs):
        r = y[0]
        return r - (Rs-Rs_sbh)
    crossing_event.terminal=True # Stops Integreation
    crossing_event.direction= -1 # Trigger when r crosses threshold from above

    return crossing_event

# "" isco
def r_isco_event(MBH, m, spin):
    r_isco=R_isco_function(MBH, spin)
    Rs_sbh=2*G*m /(c*c)
    def crossing_event(t,y, *fargs):
        r = y[0]
        return r - (r_isco-Rs_sbh)
    crossing_event.terminal=True # Stops Integreation
    crossing_event.direction= -1 # Trigger when r crosses threshold from above

    return crossing_event

def LISA_band_exit(t, y, m1, Gammas, Mbh, traps):
    r=y[0]
    GWf=GW_freq_fn(r, Mbh, m1)
    return GWf-0.1
LISA_band_exit.terminal = True
LISA_band_exit.direction = 0

def LISA_band_enter(t, y, m1, Gammas, Mbh, traps):
    r=y[0]
    GWf=GW_freq_fn(r, Mbh, m1)
    return GWf-0.001
LISA_band_enter.terminal = True
LISA_band_enter.direction = 0

#GW torque equations - from binary_formation_eqns_V8

def compute_torque_GW(disk, M, Mbh, TT):
    q = M / Mbh

    Gamma_GW = myscript.gamma_GW(disk.R, M, Mbh)

    return Gamma_GW 

def compute_GW_torque_function(disk, M, Mbh, TT):
    Gamma_tot = compute_torque_GW(disk, M, Mbh, TT)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')

def compute_noGW_torque(disk, M, Mbh, TT):
    q = M / Mbh
    
    Gamma_0 = myscript.gamma_0(q, disk.h / disk.R, 2 * disk.rho * disk.h, disk.R, disk.Omega)

    dSig = myscript.dSigmadR(disk)
    dT = myscript.dTdR(disk)
    cI_p10 = myscript.CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = myscript.CI_jm17_tot(dSig, dT, 5/3, disk)
    Gamma_I_jm17 = cI_jm17*Gamma_0

    if TT=="B16": 
        return Gamma_I_p10
    elif TT=="G23": 
        gamma = 5/3
        Gamma_therm = myscript.gamma_thermal(gamma, disk, q)*Gamma_0
        return Gamma_therm + Gamma_I_jm17

def compute_noGW_torque_function(disk, M, Mbh, TT):
    Gamma_tot = compute_noGW_torque(disk, M, Mbh, TT)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')

# Headwind Torque Eqns from Pan and Yang 2021

def drhodR(obj):
    rlog10 = np.log10(obj.R)  # descrete
    rholog10 = np.log10(obj.rho)  # descrete
    rholog10_spline = UnivariateSpline(rlog10, rholog10, k=3, ext=0)  # need scipy ver 1.10.0
    drhodR_spline = rholog10_spline.derivative()
    drhodR = drhodR_spline(rlog10)
    return drhodR

def gamma_rho(obj, mdot_gas, mbh):
    drhodr=drhodR(obj)
    h=obj.h
    cs=obj.cs
    r=obj.R
    hr= h/r
    deltav=hr*cs*(3-drhodr)/2
    gamma_wind=-r * deltav * mdot_gas * (1/mbh)
    return gamma_wind

def compute_torque_wind(disk, mdot, mbh):
    Gamma_wind = gamma_rho(disk, mdot, mbh)
    return Gamma_wind

def dJdR(obj, J):
    rlog10 = np.log10(obj.R)  # descrete
    Jlog10 = np.log10(J)  # descrete
    Jlog10_spline = UnivariateSpline(rlog10, Jlog10, k=3, ext=0)  # need scipy ver 1.10.0
    dJdR_spline = Jlog10_spline.derivative()
    dJdR = dJdR_spline(rlog10)
    return dJdR

def BHL_accretion(obj, MBH, mbh, Mdot, alpha):
    drhodr=drhodR(obj)

    M = MBH * G /(c*c)
    
    h = obj.h
    cs = obj.cs
    r = obj.R
    rho = obj.rho
    sigma = 2*h*rho
    hr = h / r

    deltav_psi=hr*cs*(3-drhodr)/2
    deltav_dr= 3/2 * (mbh/(3*MBH))**(1/3) * (1/hr) * cs

    vgas=-Mdot/(2 * np.pi * r * sigma)
    vstar=-1.3e-6 * (mbh/(10*MSun))/(MBH/(1e5*MSun)) * (r/10*M)**(-3)

    deltav_r=np.abs(vgas-vstar)
    vrel=((deltav_psi + deltav_dr)**2 + deltav_r**2)**(1/2)
    mdot_BHL= (4 * np.pi * rho * mbh * mbh) / (vrel**2 + cs**2)**(3/2)
    return mdot_BHL

def BHL_accretion2(obj, MBH, mbh, mdot, alpha):
    r=obj.R
    M=MBH * G /(c*c)

    mdot_BHL= 1.5e-7 * (alpha/0.1)**(-1) * (mdot/0.1)**(-5) * (MBH/(1e5*MSun))**(-1) * mbh**2/(10*MSun) * (r/(10*M))**6
    return mdot_BHL


#Eddington Luminosity Eqn - pAGN paper
def Ledd(MBH, X):
    kappa=0.2 * (1+X)
    Ledd= (4 * np.pi * G * MBH * c) /kappa
    return Ledd

#NTVSsg solver from disk_prfiles.ipynb, now rewritten into Novikov.py as a disk object, moved here in case it ever needs calling
def NTvsSG_disc_solver_smooth(MBH_power, spin, alpha, mdot, eps, le, steps, path, printing=True, plotting=True, save_to_file=True):
    #general scaling from Abramowicz and Fragile
    MBH=10**MBH_power * MSun #in kgs
    M=MBH * G /(c*c)
    R_G=M

    m=MBH/MSun
    Rsch= 2*G*MBH/c**2

    if mdot==None and le==None:
        raise ValueError('Please provide an accretion rate or Eddington ratio!')
    elif le==None:
        le=mdot*eps
    elif mdot==None:
        mdot=le/eps

    disk = pagn.SirkoAGN(Mbh=MBH, alpha=alpha, le=le, eps=eps)
    Rmin = disk.Rmin
    Rmax = disk.Rmax
    disk.solve_disk(N=steps)

    print(f'SG: {Rmin/R_G} Rg, {disk.Rmin/R_G} Rg, {disk.R[0]/R_G} Rg')

    Rout=Rmax
    sigma = (200 * 1e3) * (MBH / (1.3e8*MSun)) ** (1 / 4.24)
    Mdot_out = 320*(MSun/yr)*(Rout/(95*pc)) * (sigma/(188e3))**2

    diskTQM = pagn.ThompsonAGN(Mbh=MBH, Rout=Rout, Rin=Rmin, Mdot_out=Mdot_out)
    diskTQM.solve_disk()

    print(f'TQM: {Rmin/R_G} Rg, {diskTQM.Rin/R_G} Rg, {diskTQM.R[0]/R_G} Rg')

    Rmin= R_isco_function(MBH, spin) #uses relativistic eqn for ISCO to set inner edge of disc
    r_isco=R_isco_function(MBH, spin)

    R=np.logspace(np.log10(r_isco), np.log10(Rmax), steps+1)
    # R=np.linspace(r_isco, Rmax, steps+1)

    R_im=R_inner_mid(r_isco, MBH, mdot, alpha)
    R_mo=R_mid_outer(r_isco, mdot)
    R_oa=R_outer_AGN(r_isco, MBH, mdot, alpha)

    R_agn=disk.R_AGN/R_G

    if save_to_file==True:
        mypath=path

        try:
            makedirs(mypath)
        except FileExistsError:
            pass

        file = open(f'{mypath}NT_inputs.txt', "w")
        file.write(f"Input Parameters:\n")
        file.write(f"version     = V1\n")
        file.write(f"log(M_SMBH) = {MBH_power}\n")
        file.write(f"spin        = {spin}\n")
        file.write(f"alpha       = {alpha}\n")
        file.write(f"mdot        = {mdot}\n")
        file.write(f"eps         = {eps}\n")
        file.write(f"le          = {le}\n")
        file.write(f"R_min       = {Rmin:.1e}\n")
        file.write(f"R_max       = {Rmax:.1e}\n")
        file.write(f"steps       = {steps}\n")
        
    Rs=[]
    sigmas=[]
    Hs=[]
    hrs=[]
    rho0s=[]
    rhos=[]
    Ts=[]

    flag=0
    flag3=0

    r_rel=0

    inner_flag=0
    inner_transition_flag=0
    mid_flag=0
    mid_transition_flag=0
    outer_flag=0

    sf_i=0

    k=50
    if printing==True:
        print(R_oa/R_agn)
        print(f'Initial Radius = {R[k]/R_G} Rg, R_isco = {r_isco/R_G} Rg')

    for i in range(k, steps+1):
        r=R[i]
        y=np.sqrt(r/M)
        rstar=(r/M)

        #######TO EDIT

        if rstar<0.5*R_im:
            if inner_flag==0:
                if printing==True:
                    print(f'disk confidently in inner region')
                inner_flag=1
            rho_0=rho_0_NT(r, MBH, spin, mdot, alpha)
            T=T_NT(r, MBH, spin, mdot, alpha)
            H=H_NT_2(r, MBH, spin, mdot)
            sigma=Sigma_NT(r, MBH, spin, mdot, alpha)

        if 0.5*R_im<=rstar<5*R_im:
            if inner_transition_flag==0:
                if printing==True:
                    print(f'disk in inner-mid transition region')
                inner_transition_flag=1
            
            param_in=Sigma_NT(r, MBH, spin, mdot, alpha)
            param_mid=Sigma_NT_Middle(r, MBH, spin, mdot, alpha)

            if param_in-param_mid>0:
                if mid_flag==0:
                    if printing==True:
                        print(f'disk transitions to middle region at {rstar} Rg')
                        r_im=rstar
                    mid_flag=1
                rho_0=rho_0_NT_Middle(r, MBH, spin, mdot, alpha)
                T=T_NT_Middle(r, MBH, spin, mdot, alpha)
                H=H_NT_Middle(r, MBH, spin, mdot, alpha)
                sigma=Sigma_NT_Middle(r, MBH, spin, mdot, alpha)
            else:
                sigma=Sigma_NT(r, MBH, spin, mdot, alpha)
                rho_0=rho_0_NT(r, MBH, spin, mdot, alpha)
                T=T_NT(r, MBH, spin, mdot, alpha)
                H=H_NT_2(r, MBH, spin, mdot)
            
        
        if 5*R_im<=rstar<0.5*R_mo:
            rho_0=rho_0_NT_Middle(r, MBH, spin, mdot, alpha)
            T=T_NT_Middle(r, MBH, spin, mdot, alpha)
            H=H_NT_Middle(r, MBH, spin, mdot, alpha)
            sigma=Sigma_NT_Middle(r, MBH, spin, mdot, alpha)

        if 0.5*R_mo<=rstar<100*R_mo:
            if mid_transition_flag==0:
                if printing==True:
                    print(f'disk in mid-outer transition region')
                mid_transition_flag=1
            
            param_out=Sigma_NT_Outer(r, MBH, spin, mdot, alpha)
            param_mid=Sigma_NT_Middle(r, MBH, spin, mdot, alpha)

            if param_mid-param_out>0:
                if outer_flag==0:
                    if printing==True:
                        print(f'disk transitions to outer region at {rstar} Rg')
                        r_mo=rstar
                    outer_flag=1
                rho_0=rho_0_NT_Outer(r, MBH, spin, mdot, alpha)
                T=T_NT_Outer(r, MBH, spin, mdot, alpha)
                H=H_NT_Outer(r, MBH, spin, mdot, alpha)
                sigma=Sigma_NT_Outer(r, MBH, spin, mdot, alpha)
            else:
                rho_0=rho_0_NT_Middle(r, MBH, spin, mdot, alpha)
                T=T_NT_Middle(r, MBH, spin, mdot, alpha)
                H=H_NT_Middle(r, MBH, spin, mdot, alpha)
                sigma=Sigma_NT_Middle(r, MBH, spin, mdot, alpha)
        
        if 5*R_mo<rstar:
            rho_0=rho_0_NT_Outer(r, MBH, spin, mdot, alpha)
            T=T_NT_Outer(r, MBH, spin, mdot, alpha)
            H=H_NT_Outer(r, MBH, spin, mdot, alpha)
            sigma=Sigma_NT_Outer(r, MBH, spin, mdot, alpha)

        ########END
        
        sigmas.append(sigma)
        Ts.append(T)
        Hs.append(H)
        rho0s.append(rho_0)
        Rs.append(rstar)

        hr=H/(100*r)

        hrs.append(hr)

        rho=sigma/(2*H)
        rhos.append(rho)

        omega=np.sqrt(G * MBH / (r*r*r))

        v=omega * r
        vc=v/c

        Qt= omega*omega / (2 * np.pi * G * rho)

        if vc<0.1 and flag==0:
            if printing==True:
                print(f'disk stops being relativistic at {r/R_G} Rg')
            r_rel=r
            flag+=1
        
        if r>=disk.R_AGN and flag3==0:
            if printing==True:
                print(f'disk begins star formation at {r/R_G} Rg')
            sf_i=i
            r_outer=r
            flag3+=1
        
    if save_to_file==True:
        file = open(f'{mypath}NT_inputs_smooth.txt', "a")
        file.write(f"SF          = {flag3}\n")
        file.write(f"SF index    = {sf_i}\n")
        d=dict({'Radius [Rg]': Rs, 'Surface Density [gcm^-2]': sigmas, 'Temperature [K]': Ts, 'Midplane Density [gcm^-2]': rho0s, 'Thickness [cm]': Hs, 'Aspect Ratio': hrs})
        df=pd.DataFrame.from_dict(d)
        df.to_csv(f'{mypath}NT_disc_smooth.csv', index=False)

    if plotting==True:

        fig, axs = plt.subplots(1, 5, figsize=(25, 5), dpi=100)

        colour='plasma'
        cmap = plt.colormaps[colour]

        plt.suptitle(f'$SMBH = 10^{MBH_power}'r'{M_{\odot}}, \alpha$ = 'f'{alpha},'r'$\chi$ = 'f'{spin}')

        i=0

        axs[i].plot(disk.R/M, 2*disk.h*disk.rho*SI_to_gcm2, label = r"SG", color=cmap(0.0))
        axs[i].plot(diskTQM.R/M, 2*diskTQM.h*diskTQM.rho*SI_to_gcm2, label = r"TQM", color=cmap(0.5))
        axs[i].plot(Rs, sigmas, '-', color=cmap(0.8), label = r"NT")

        axs[i].set_ylabel(r'$\Sigma_{\rm g} [{\rm g \, cm}^{-2}]$')

        i=1

        axs[i].plot(disk.R/M, disk.T, color=cmap(0.0), label = r"SG")
        axs[i].plot(diskTQM.R/M, diskTQM.T, color=cmap(0.5), label = r"TQM")
        axs[i].plot(Rs, Ts, '-', color=cmap(0.8), label = r"NT")

        axs[i].set_ylabel(r'$T [K] $')

        i=2

        axs[i].plot(disk.R/M, disk.rho*SI_to_gcm3, color=cmap(0.0), label = r"SG")
        axs[i].plot(diskTQM.R/M, diskTQM.rho*SI_to_gcm3, color=cmap(0.5), label = r"TQM")
        axs[i].plot(Rs, rho0s, '-', color=cmap(0.8), label = r"NT")

        axs[i].set_ylabel(r'$\rho [gcm^{-3}] $')

        i=3

        axs[i].plot(disk.R/M, disk.h*SI_to_cms, '-', color=cmap(0.0), label = r"SG")
        axs[i].plot(diskTQM.R/M, diskTQM.h*SI_to_cms, '-', color=cmap(0.5), label = r"TQM")
        axs[i].plot(Rs, Hs, '-', color=cmap(0.8), label = r"NT")

        axs[i].set_ylabel(r'$H [cm]$')
        axs[i].legend()

        i=4

        axs[i].plot(disk.R/M, disk.h/disk.R, '-', color=cmap(0.0), label = r"SG")
        axs[i].plot(diskTQM.R/M, diskTQM.h/diskTQM.R, '-', color=cmap(0.5), label = r"TQM")
        axs[i].plot(Rs, hrs, '-', color=cmap(0.8), label = r"NT")

        axs[i].set_ylabel(r'$H/R$')

        for i in range(0, 5):
            axs[i].axvline(x=r_im, linestyle='--', color=cmap(0.9), alpha=0.5, label = r"$R_{inner}$")
            axs[i].axvline(x=r_rel/M, linestyle='--', color=cmap(0.7), alpha=0.5, label = r"$R_{rel}$")
            axs[i].axvline(x=r_mo, linestyle='--', color=cmap(0.4), alpha=0.5, label = r"$R_{outer}$")
            axs[i].axvline(x=disk.R_AGN/M, linestyle='--', color=cmap(0.2), alpha=0.5, label = r"$R_{AGN}$")

            axs[i].set_xscale('log')
            axs[i].set_yscale('log')

            axs[i].set_xlabel("r/M")
            axs[i].set_xlim(1e0, 3e7)

        axs[i].legend()

        plt.tight_layout()
        if save_to_file==True:
            plt.savefig(f'{mypath}all_profiles_with_TQM_smooth.pdf')
        plt.show()



def T_align(disk, Mbh, mbh, cos_i, H, R):
    i=np.arccos(cos_i)
    cos_i2=np.cos(i/2)
    sin_i2=np.sin(i/2)
    t_orb=2*np.pi*(R)**(2/3) * (G * Mbh)**(-1/2)
    t_align= (t_orb * Mbh**2)/(2*mbh*disk.Mdisk) * cos_i2 * (sin_i2**2 + H/(4*R))**2
    return t_align

def cluster_df(cluster, R, cos_i, disk):
    # print(f'cluster:{cluster}\n R:{R}\n cosi:{cos_i}\n disk:{disk}')
    Mbh=disk.Mbh

    R_g=Mbh * G /(c*c)

    d = {"mbh [Msun]": cluster, 'r [Rg]': R/R_g, 'cos_i': cos_i}
    df=pd.DataFrame(data=d)

    f=interp1d(disk.R, disk.h, kind='linear', fill_value='extrapolate')
    h_clust=f(df['r [Rg]'] * R_g)

    df["H/R"]=h_clust/(df['r [Rg]'] * R_g)
    df['H']=h_clust

    mbh=df['mbh [Msun]']*MSun

    i=np.arccos(cos_i)
    cos_i2=np.cos(i/2)
    sin_i2=np.sin(i/2)

    t_orb=2*np.pi*(df['r [Rg]'] * R_g)**(2/3) * (G * Mbh)**(-1/2)
    # t_align= (t_orb * Mbh**2)/(2*mbh*disk.Mdisk) * cos_i2 * (sin_i2**2 + df['H']/(4*df['r [Rg]']*R_g))**2
    t_align=T_align(disk, Mbh, mbh, df['cos_i'], df['H'], df['r [Rg]'])

    df['t_align [yrs]']=t_align/(365*24*60*60)

    p=1-np.exp((-(1e7*365*24*60*60)/(t_align)))
    df['p_align']=p

    return df

def cluster_sampling(MBH_digit, MBH_power, alpha, eps, spin, le, DT, BIMF, save):

    Mbh=Mbh=MBH_digit * 10**MBH_power * MSun

    Ledd=Ledd(Mbh, X=0.7)
    Mdot_edd = Ledd / (eps.c ** 2)
    Mdot = le * Mdot_edd

    # mdot=Mdot/Mdot_edd
    # print(mdot)

    if DT=="SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha, le=le, eps=eps)
        # disk = pagn.SirkoAGN(Mbh=Mbh)
        Rmin = disk.Rmin
        Rmax = disk.Rmax
        disk.solve_disk()

    R_g=Mbh * G /(c*c)

    Rmax=0.1 * pc * (Mbh/(1e6 * MSun))**(1/2)

    cluster=[]
    cluster_mass=2 * Mbh/MSun

    if BIMF=='Tagawa':
    #Tagawa et al 2020 BIMF
        cluster_tagawa = imf.make_cluster(cluster_mass, massfunc='salpeter', alpha=2.3, mmin=0.1, mmax=140)
        for mass in cluster_tagawa:
            if mass<20:
                continue
            elif 20<=mass<40:
                mass_bh=mass/4
            elif 40<=mass<=55:
                mass_bh=10
            elif 55<=mass<=120:
                mass_bh=mass/13 + 5.77
            else:
                mass_bh=15
            cluster.append(mass_bh)
        bh_mass_tot=np.sum(cluster)
        print(f'Total bh mass is {bh_mass_tot}')

    elif BIMF=='Bartos':
    #Bartos et al 2017 BIMF
        bartos_cluster=imf.make_cluster(0.04*cluster_mass, massfunc='salpeter', alpha=2, mmin=5, mmax=50)
        cluster=bartos_cluster
        print(f'Total bh mass is {np.sum(cluster)}')

    elif BIMF=='Vaccaro':
    #Actually from Iorio et al 2023, but file provided by MP Vaccaro
        mass_tot=0
        mass_sec=np.genfromtxt("BHs_single_Zsun_rapid_nospin.dat",usecols=(0),skip_header=3,unpack=True)
        while mass_tot<0.04*cluster_mass:
            c = np.random.randint(0, len(mass_sec))
            cluster.append(mass_sec[c])
            mass_tot+=mass_sec[c]
        print(f'Total bh mass is {np.sum(cluster)}')

    a=powerlaw.Power_Law(alpha=2.5, xmin=12*R_g, xmax=Rmax)
        # print(np.max(a.rvs(len(iorio_bhs))))
    R=a.generate_random(len(cluster))

    cos_i=np.random.uniform(-1.0, 1.0, len(cluster))
    df=cluster_df(cluster, R, cos_i, disk)

    if save==True:
        df.to_csv(f'EMRI_Rates/{BIMF}/dataframes/{DT}_{MBH_digit}e{MBH_power}_alpha_{alpha}_eps_{eps}_le_{le}_spin_{spin}.csv')
    return df

def plot_cluster(df, MBH_digit, MBH_power, alpha, eps, le, spin, BIMF, t_agn, DT, save=False):
    plt.figure(figsize=(6, 8))
    plt.scatter(df['r [Rg]'], df['mbh [Msun]'], c=df['t_align [yrs]']/t_agn, cmap='gist_rainbow', norm='log')
    plt.xscale('log')
    plt.colorbar(label="$t_{align}/t_{AGN}$", orientation="horizontal")
    # plt.clim(0,1)
    plt.xscale('log')
    plt.xlabel(r'$R~[R_g]$')
    plt.ylabel(r'$mbh~[M_{\odot}]$')
    plt.title('$t_{align}/t_{AGN},~$' f'SMBH={MBH_digit}e{MBH_power}'r'$M_{\odot}, ~\alpha$' f'$={alpha},~e={eps},~l_e={le},~X={spin}$')
    if save==True:
        plt.savefig(f'EMRI_Rates/{BIMF}/t_align_{DT}_{MBH_digit}e{MBH_power}_alpha_{alpha}_eps_{eps}_le_{le}_spin_{spin}.png')
    plt.show()

    plt.figure(figsize=(6, 8))
    plt.scatter(df['r [Rg]'], df['mbh [Msun]'], c=df['p_align'], cmap='gist_rainbow')
    plt.xscale('log')
    plt.colorbar(label="$p_{align}$", orientation="horizontal")
    plt.clim(0,1)
    plt.xscale('log')
    plt.xlabel(r'$R~[R_g]$')
    plt.ylabel(r'$mbh~[M_{\odot}]$')
    plt.title('$p_{align},~$' f'SMBH={MBH_digit}e{MBH_power}'r'$M_{\odot}, ~\alpha$' f'$={alpha},~e={eps},~l_e={le},~X={spin}$')
    if save==True:
        plt.savefig(f'EMRI_Rates/{BIMF}/p_align_{DT}_{MBH_digit}e{MBH_power}_alpha_{alpha}_eps_{eps}_le_{le}_spin_{spin}.png')
    plt.show()
