import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import binary_formation_distribution_V8 as myscript
import binary_formation_distribution_V11 as myscript2
from scipy.interpolate import interp1d, UnivariateSpline

from scipy.stats import rv_continuous

import imf 
import powerlaw
import pandas as pd

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

print('You have ran the wrong file lol')

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
def Sigma_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def Sigma_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def Sigma_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def H_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot

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

def H_NT_2(r, disk):
    #added factor of m like in Grishin et al 2025, seems to fix a scaling issue
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot

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

def H_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def H_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def T_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    alpha=disk.alpha

    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    A=A_fn(y, spin)
    B=B_fn(y, spin)
    E=E_fn(y, spin)

    T= (5e7) * alpha**(-1/4) * m**(-1/4) * rstar**(-3/8) * A**(-1/2) * B**(1/2) * E**(1/4) #in Kelvin
    return T

def T_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    B=B_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    T = (7e8) * alpha**(-1/5) * m**(-1/5) * mdot**(2/5) * rstar**(-9/10) * B**(-2/5) * D**(-1/5) * Q**(2/5)
    return(T)

def T_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def rho_0_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def rho_0_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def rho_0_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def F_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def Beta_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def Beta_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def Beta_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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
def Tau_NT(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha

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

def Tau_NT_Middle(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot

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

def Tau_NT_Outer(r, disk):
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot

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

def Template_NT(r, disk):
    # Template NT function to make coding the rest of the equations easier
    MBH=disk.Mbh
    spin=disk.spin
    mdot=disk.mdot
    alpha=disk.alpha
    
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

def R_inner_mid(self):
    r=self.Rmin
    MBH=self.Mbh 
    mdot=self.mdot
    alpha=self.alpha

    m=MBH/(1e8*MSun)
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=449.842 * alpha**(2/21) * m**(2/21) * mdotprime**(16/21)
    return(r_tr)

def R_mid_outer(self):
    r=self.Rmin
    mdot=self.mdot
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=987.891 * mdotprime**(2/3)
    return(r_tr)

def R_outer_AGN(self):
    r=self.Rmin
    MBH=self.Mbh 
    mdot=self.mdot
    alpha=self.alpha
    m=MBH/(1e8*MSun)
    mdotprime=mdot*(1-r**(-1/2))
    r_tr=580.65 * alpha**(28/45) * m**(-52/45) * mdotprime**(-22/45)
    return(r_tr)

#Eqns derived from A+F NT Surface Density Profile Eqns - didn't work

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

    dSig = myscript.dSigmadR(disk)
    dT = myscript.dTdR(disk)
    cI_p10 = myscript.CI_p10(disk, dSig, dT)
    Gamma_I_p10 = cI_p10*Gamma_0
    cI_jm17 = myscript.CI_jm17_tot(dSig, dT, 5/3, disk)
    Gamma_I_jm17 = cI_jm17*Gamma_0

    if args.TT=="B16": 
        return Gamma_I_p10
    elif args.TT=="G23": 
        gamma = 5/3
        Gamma_therm = myscript.gamma_thermal(gamma, disk, q)*Gamma_0
        return Gamma_therm + Gamma_I_jm17

def compute_noGW_torque_function(args, disk, M, Mbh):
    Gamma_tot = compute_noGW_torque(args, disk, M, Mbh)
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

def BHL_accretion(args, obj, MBH, mbh, Mdot):
    drhodr=drhodR(obj)

    M = MBH * G /(c*c)
    
    h = obj.h
    cs = obj.cs
    r = obj.R
    rho = obj.rho
    sigma = 2*h*rho
    hr = h / r

    deltav_psi=hr*cs*(3-drhodr)/2
    deltav_dr=3/2 * (mbh/(3*MBH))**(1/3) * (1/hr) * cs

    vgas=-Mdot/(2 * np.pi * r * sigma)
    vstar=-1.3e-6 * (mbh/(10*MSun))/(MBH/(1e5*MSun)) * (r/10*M)**(-3)

    deltav_r=np.abs(vgas-vstar)
    vrel=((deltav_psi + deltav_dr)**2 + deltav_r**2)**(1/2)
    mdot_BHL= (4 * np.pi * rho * mbh * mbh) / (vrel**2 + cs**2)**(3/2)
    return mdot_BHL

def BHL_accretion2(args, obj, MBH, mbh, mdot):
    alpha=args.a
    r=obj.R
    M=MBH * G /(c*c)

    mdot_BHL= 1.5e-7 * (alpha/0.1)**(-1) * (mdot/0.1)**(-5) * (MBH/(1e5*MSun))**(-1) * mbh**2/(10*MSun) * (r/(10*M))**6
    return mdot_BHL


#Eddington Luminosity Eqn - pAGN paper
def Ledd(MBH, X):
    kappa=0.2 * (1+X)
    Ledd= (4 * np.pi * G * MBH * c) /kappa
    return Ledd

def T_align(disk, Mbh, mbh, cos_i, H, R):
    i=np.arccos(cos_i)
    cos_i2=np.cos(i/2)
    sin_i2=np.sin(i/2)
    t_orb=2*np.pi*(R)**(2/3) * (G * Mbh)**(-1/2)

    # print(f'torb: {t_orb}, Mbh: {Mbh}, mbh: {mbh}, mdisk: {disk.Mdisk}, h: {H}, R: {R}')
    t_align= (t_orb * Mbh**2)/(2*mbh*disk.Mdisk) * cos_i2 * (sin_i2**2 + H/(4*R))**2
    # print(f't_align: {t_align/(365*24*60*60*1e6)} Myr')
    return t_align

def T_enc(MBH, mbh, R, N, disk):
    Rmax=disk.Rmax
    rh = R * (mbh / (3. * MBH))**(1./3.)
    R_g=MBH * G /(c*c)
    omega=np.sqrt(G*MBH/(R*R*R))

    f=interp1d(disk.R, disk.h, kind='linear', fill_value='extrapolate')
    h_clust=f(R * R_g)

    vrel=3/2 * omega * rh
    nbh=N/(4/3 * np.pi * Rmax**3)
    zh=np.minimum(rh, h_clust)

    t=1/(nbh * rh * zh * vrel)

    # print(f't_enc: {t/(365*24*60*60*1e6)} Myr')
    return t

def R_in(Mbh, mbh, T):
    M=Mbh+mbh
    Mm=Mbh*mbh

    a=(256/5 * (1/c**5) * G**3 * M * Mm * T)**(1/4)
    return a

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

def R_I_fn(MBH, mbh, mstar, N):
    Nstar = 2*MBH/mstar

    sig = (MBH/(3.097*10**8*MSun))**(1/4) * (200 * 1000)
    f=N/Nstar
    m_ratio=mstar/mbh

    Rh = G*MBH/(sig**2)
    R_I = f**(4/5) * m_ratio**(-6/5) * Rh
    return R_I

def R_II_fn(MBH, mbh, mstar, N):
    Rs = 2 * G * MBH/(c**2)
    Nstar = 2*MBH/mstar

    f=N/Nstar
    m_ratio=mbh/mstar
    fc=4.5e-4*(m_ratio/10)

    R_II  = 4*Rs*(fc/f)**(2/5)

    if R_II<4*Rs:
        R_II=4*Rs
    return R_II

def R_GW_fn(MBH, mbh, mstar, N):
    Rs = 2 * G * MBH/(c**2)
    sig = (MBH/(3.097*10**8*MSun))**(1/4) * (200 * 1000)

    alpha = 425 * np.pi**2 / (2048 * 2**(1/2) * 1.35 * 10) # Bortolas and Mapelli 2019
    cp = 1.13 #Kaur et al 2024
    m_ratio=mstar/mbh

    Rh = G*MBH/(sig**2)

    R_GW = Rs * (10 * alpha**(1/5) / (cp**2))**(4/3) * (m_ratio)**(-2/15) * (Rs/Rh)**(-1/3)
    return R_GW

class RomDistribution(rv_continuous):
    def __init__(self, MBH, mbh, mstar, N):
        self.MBH = MBH
        self.mbh = mbh
        self.mstar = mstar
        self.N = N

    def _pdf(self, r, approx=False):
        MMW=4e6*MSun
        MBH=self.MBH
        mbh=self.mbh
        mstar=self.mstar
        N=self.N

        Rs = 2 * G * MBH/(c**2)
        
        sig = (MBH/(3.097*10**8*MSun))**(1/4) * (200 * 1000) #M-𝜎 relation Ferrarese and Merrit 2000

        Nstar = 2*MBH/mstar
        alpha = 425 * np.pi**2 / (2048 * 2**(1/2) * 1.35 * 10) # Bortolas and Mapelli 2019
        cp = 1.13 #Kaur et al 2024

        f=N/Nstar
        # print(f'f: {f}')
        m_ratio=mbh/mstar
        fc=4.5e-4*(m_ratio/10)

        C1 = cp**2 / (alpha**(1/5) * 10)
        C2 = (fc/f)**(4/5) 
        if C2>0.9:
            C2=0.9

        Rh = G*MBH/(sig**2)
        R_I = f**(4/5) * m_ratio**(6/5) * Rh
        R_II  = 4*Rs*(fc/f)**(2/5)
        if R_II<4*Rs:
            R_II=4*Rs
        R_GW = Rs * (10 * alpha**(1/5) / (cp**2))**(4/3) * (m_ratio)**(2/15) * (Rs/Rh)**(-1/3)
        const = 1/(2*np.pi*Rh**3) * (MBH/mstar)

        if approx==True:
            print("APPROX")
            alpha=0.11
            Rs= 4e-7 * pc * (MBH/(MMW))
            Rh=2*pc*(MBH/(MMW))
            R_GW = Rs * 2e3 * (MBH/MMW)
        
        # print(f'r: {r/Rs:.1e} Rs, Rh: {Rh/Rs:.1e} Rs, R_I: {R_I/Rs:.1e} Rs, R_GW: {R_GW/Rs:.1e} Rs, R_II: {R_II/Rs:.1e} Rs\n')
        
        if R_I < r < Rh:
            # print(f'Outer Region, r: {r/Rs:.2e} Rs')
            n = const * f**(9/5) * (m_ratio)**(6/5) * (r/Rh)**(-4)
        elif R_GW < r < R_I:
            # print(f'Outside GW scheme, r: {r/Rs:.2e} Rs')
            n = const * (m_ratio)**(-3/2) * (r/Rh)**(-7/4)
        elif R_II < r < R_GW:
            # print(f'Inside GW scheme, r: {r/Rs:.2e} Rs')
            n = const * C1 * (m_ratio)**(-8/5) * (Rs/Rh)**(-3/2) * (r/Rs)**(-1)
        elif Rs <= r < R_II:
            # print(f'Inner Region, r: {r/Rs:.2e} Rs')
            n = const * C2 * f**(4/5) * (m_ratio)**(-4/5) * (Rs/Rh)**(-3/2) * (r/Rs)
        return n*r*r*np.pi*4
    

def rejection_sample(fn, minimum, maximum, M, N):
    samples=[]
    while len(samples)<N:
        log_uniform_y=rand.uniform(np.log10(minimum), np.log10(maximum))
        uniform_y=10**log_uniform_y
        uniform_u=rand.uniform(0,1)
        # print(uniform_u, uniform_y)
        if uniform_u*M<fn(uniform_y):
            # print(f'uniform_u={uniform_u} accepted')
            samples.append(uniform_y)
    return samples[:N]

def plot_loghist(x, bins, axes, **kwargs):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    axes.hist(x, bins=logbins, **kwargs)
    return np.max(hist)

def cluster_sampling(MBH, alpha, spin, le, DT, BIMF, RD, disk, T, gamma, save=True):
    Mbh=MBH
    power=int(np.log10(MBH/MSun))
    digit=Mbh/(MSun * 10**power)

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
            print(f'populating cluster... {len(cluster)}', end='\r')
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
            n = np.random.randint(0, len(mass_sec))
            cluster.append(mass_sec[n])
            print(f'populating cluster... {len(cluster)}', end='\r')
            mass_tot+=mass_sec[n]
        print(f'Total bh mass is {np.sum(cluster)}')

    elif BIMF=='PY':
        #Pan and Yang 2021
        Mstar=20*Mbh
        delta=1e-3
        Nstar=Mstar/(1*MSun)
        Nbh=delta*Nstar
        for i in range(0, int(Nbh)):
            cluster.append(10)
        print(f'Total bh mass is {np.sum(cluster)}')

    if RD=='Bartko':
        R_min = R_in(Mbh, np.mean(cluster) * MSun, T)
        print(f'R_clust: {R_min/R_g} Rg, {R_min/pc} pc')
        a=powerlaw.Power_Law(alpha=gamma+2, xmin=R_min, xmax=Rmax)
            # print(np.max(a.rvs(len(iorio_bhs))))
        R=a.generate_random(len(cluster))
    if RD=='PY':
        R_min = 6*R_g
        print(f'R_clust: {R_min/R_g} Rg, {R_min/pc} pc')
        a=powerlaw.Power_Law(alpha=gamma+2, xmin=R_min, xmax=Rmax)
            # print(np.max(a.rvs(len(iorio_bhs))))
        R=a.generate_random(len(cluster))
    elif RD=='Rom':
        mbh=10*MSun
        mstar=1*MSun
        a=RomDistribution(MBH=Mbh, mbh=mbh, mstar=mstar, N=N)
        sig = (Mbh/(3.097*10**8*MSun))**(1/4) * (200 * 1000)
        Rh = G*Mbh/(sig**2)
        R_I=R_I_fn(Mbh, mbh, mstar,len(cluster))
        if R_I<=Rh:
            nmax=a._pdf(R_I, approx=False)
        elif R_I>Rh:
            nmax=a._pdf(Rh, approx=False)
        R=rejection_sample(a._pdf, 2*R_g, Rh, nmax, len(cluster))

    cos_i=np.random.uniform(-1.0, 1.0, len(cluster))
    df=cluster_df(cluster, R, cos_i, disk)

    if save==True:
        df.to_csv(f'EMRI_Rates/{BIMF}/dataframes/{DT}_{digit}e{power}_alpha_{alpha}_le_{le}_spin_{spin}_N_{len(cluster)}.csv')
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
        plt.savefig(f'EMRI_Rates/{BIMF}/p_align_{DT}_{MBH_digit}e{MBH_power}_alpha_{alpha}_eps_{eps}_le_{le}_spin_{spin}_3.png')
    plt.show()

def plot_torques(args, disk, Mbh, mass_sec, T):
    Rsch = 2 * G * Mbh / c**2
    Rs = disk.R / Rsch

    Mmean=np.mean(mass_sec)* MSun 

    mean_Gamma=myscript2.compute_torque(disk, float(Mmean), Mbh, args.TT, args.wind) 

    traps = myscript2.mig_trap(disk, mean_Gamma) 
    traps=np.array(traps)/Rsch

    antitraps = myscript2.anti_trap(disk, mean_Gamma) 
    antitraps=np.array(antitraps)/Rsch

    R_min = R_in(Mbh, Mmean, T)/Rsch
    # R_min = 3*Rsch

    fig, ax = plt.subplots(figsize=(8, 6))

    for trap in traps:
        ax.axvline(trap *2, color='gray', linestyle=':', linewidth=1, label='Migration trap' if trap == traps[0] else "")
    for trap in antitraps:
        ax.axvline(trap *2, color='gray', linestyle='--', linewidth=1, label='Anti-trap'if trap == antitraps[0] else "")

    ax.axvline(R_min *2, color='gray', linestyle='-', linewidth=1, label='$R_{in, cluster}$')

    # Mean line: solid for Gamma > 0, dashed for Gamma < 0
    # Split into continuous regions of positive or negative torque

    sign_changes = np.where(np.diff(np.sign(mean_Gamma)) != 0)[0]
    split_indices = np.concatenate(([0], sign_changes + 1, [len(Rs)]))

    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        R_seg = Rs[start:end] *2
        G_seg = mean_Gamma[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='$|\Gamma_{tot}| >0$' if i == 1 else "", color='rebeccapurple')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='$|\Gamma_{tot}| <0$' if i == 0 else "", color='rebeccapurple')

    power=round(np.log10(Mbh/MSun))

    max_gamma=round(np.log10(np.max(mean_Gamma)))+1
    min_gamma=max_gamma-10

    print(max_gamma, min_gamma)

    min_gamma=10**min_gamma
    max_gamma=10**max_gamma

    print(f'{max_gamma:1e}, {min_gamma:1e}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'R [R$_{\rm g}$]')
    ax.set_ylabel(r'$|\Gamma|$ [cgs]')
    ax.set_ylim(1e32,1e42)
    ax.set_title(f'Migration Torques ($SMBH = 10^{power}$'r'${M_{\odot}})$')
    ax.legend()
    plt.tight_layout()
    if args.DT  == "TQM":
        plt.savefig(f'Torques/TQM/Antitraps_Mbh_{np.log10(Mbh/MSun):.1f}_{args.DT}_{args.TT}_{args.gen}_wind_MP.pdf', format='pdf', dpi=300)
    elif args.DT == "SG" or "NT":
        path=f'Torques/{args.DT}/Antitraps_Mbh_{np.log10(Mbh/MSun):.2f}_alpha_{args.a}_{args.DT}_{args.TT}_{args.gen}_3.pdf'
        print(f'Torque plot saves to {path}')
        plt.savefig(f'{path}', format='pdf', dpi=300)
    return
