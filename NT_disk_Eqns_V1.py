import numpy as np

import matplotlib.pyplot as plt
import multiprocessing
import binary_formation_distribution_V8 as myscript
from scipy.interpolate import interp1d

c = 2.99792458e8  # m/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
MSun = 1.98847e30  # kg

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

def H_NT(r, MBH, spin, mdot):
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

def compute_torque_GW(args, disk, M, Mbh):
    q = M / Mbh

    Gamma_GW = myscript.gamma_GW(disk.R, M, Mbh)

    if args.TT=="B16": 
        return Gamma_GW
    elif args.TT=="G23": 
        gamma = 5/3
        return Gamma_GW 

def compute_GW_torque_function(args, disk, M, Mbh):
    Gamma_tot = compute_torque_GW(args, disk, M, Mbh)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')

def compute_noGW_torque(args, disk, M, Mbh):
    q = M / Mbh
    
    Gamma_0 = myscript.gamma_0(q, disk.h / disk.R, 2 * disk.rho * disk.h, disk.R, disk.Omega)
    Gamma_GW = myscript.gamma_GW(disk.R, M, Mbh)

    dSig = myscript.dSigmadR(disk)
    dT = myscript.dTdR(disk)
    cI_p10 = myscript.CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = myscript.CI_jm17_tot(dSig, dT, 5/3, disk)
    Gamma_I_jm17 = cI_jm17*Gamma_0

    if args.TT=="B16": 
        return Gamma_I_p10 #+ Gamma_GW
    elif args.TT=="G23": 
        gamma = 5/3
        Gamma_therm = myscript.gamma_thermal(gamma, disk, q)*Gamma_0
        return Gamma_therm + Gamma_I_jm17 #+ Gamma_I_p10 + Gamma_GW

def compute_noGW_torque_function(args, disk, M, Mbh):
    Gamma_tot = compute_noGW_torque(args, disk, M, Mbh)
    return interp1d(disk.R, Gamma_tot, kind='linear', fill_value='extrapolate')
