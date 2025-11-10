import numpy as np

import matplotlib.pyplot as plt
import multiprocessing

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

def H_NT(r, MBH, spin, mdot):
    M=MBH * G /(c*c)
    y=np.sqrt(r/M)
    
    A=A_fn(y, spin)
    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    E=E_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    H=1e5 * mdot * A**2 * B**(-3) * C**(1/2) * D**(-1) * E**(-1) * Q #in cms
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

def Sigma_NT_Middle(r, MBH, spin, mdot, alpha):
    M=MBH * G /(c*c)
    m=MBH/MSun
    rstar=(r/M)
    y=np.sqrt(r/M)

    B=B_fn(y, spin)
    C=C_fn(y, spin)
    D=D_fn(y, spin)
    Q=Q_fn(y, MBH, spin)

    sigma = (9e4) * alpha**(-4/5) * m**(1/5) * mdot**(3/5) * rstar**(-3/5) * B**(-4/5) ** C**(1/2) * D**(-4/5) * Q**(3/5)
    return sigma

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