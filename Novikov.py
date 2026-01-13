import numpy as np
import pagn.constants as ct
import matplotlib.pyplot as plt
import pagn
import NT_disk_Eqns_V2 as jscript
from os import makedirs

import pandas as pd

class NovikovThorneAGN:
    def __init__(self, Mbh=1e8*ct.MSun, spin=0.9, alpha=0.01, mdot=0.1, eps=0.1, le=None, b=0,
                X=0.7, printing=True):
        """ Class that creates an AGN disc object using the equations for Novikov-Thorne disk (1973) from Abramowicz and Fragile 2003.

        Parameters
        ----------
        Mbh: float, optional (default: 1e8*ct.MSun)
            Mass of the Super Massive Black Hole (SMBH) fuelling the AGN in kg.
        spin: float, optional (default: 0.9)
            Spin of the SMBH fuelling the AGN.
        alpha: float, optional (default: 0.01)
            Viscosity parameter of the inner region of the AGN.
        mdot: float, optional (default: 0.1)
            Mass accretion rate, taken as constant throughout the disc, in units of ----
        eps: float, optional (default: 0.1)
            SMBH radiative efficiency parameter.
        le: float, optional (default: None)
            Luminosity ratio L0/LE, where L0 is the non self gravitating luminosity and LE is the Eddington luminosity of the SMBH.
        b: float, optional (default: 0.)
            Power index for viscosity-gas pressure relation, can only be 0 or 1.
        Mdot: float, optional (default: None)
            Mass accretion rate, taken as constant throughout the disc, in units of kg/s. Can be given instead of le.
        X: float, optional (default: 0.7)
            Hydrogen abundance of gas in AGN disk.

        """

        self.Mbh = Mbh
        self.alpha = alpha
        self.b = b
        self.spin = spin

        M=Mbh * ct.G /(ct.c*ct.c)

        self.Rg=M

        self.eps = eps
        self.X = X

        if mdot==None and le==None:
            raise ValueError('Please provide an accretion rate or Eddington ratio!')
        elif le==None:
            le=mdot*eps
        elif mdot==None:
            mdot=le/eps
        
        self.mdot=mdot
        self.le=le

        self.Rs = 2 * self.Mbh * ct.G / (ct.c ** 2)
        self.Rmin = jscript.R_isco_function(self.Mbh, self.spin)

        self.Rmax = 1e7 * self.Rs

        if printing==True:
            print("### Novikov-Thorne 1973 parameters ###")
            print("Mbh = {:e} MSun".format(self.Mbh / ct.MSun))
            print("spin = ", self.spin)
            print("mdot = ", self.mdot)
            print("eps =", self.eps)
            print("le =", self.le)
            print("Rg = {:e} pc".format(self.Rg / ct.pc))
            print("Rmin = {:e} Rg".format(self.Rmin / self.Rg))
            print("Rmax = {:e} Rg, {:e} pc".format(self.Rmax / self.Rg, self.Rmax / ct.pc))
            print("alpha =", self.alpha)
            print("b =", self.b)
            print("X =", self.X)
    

    def solve_disk(self, steps=1000, path=None,  printing=True, save_to_file=False):
        """ Method to evolve the AGN disc, from outer boundary inwards, using Novikov Thorne ---- equations.

        Parameters
        ----------
        steps: float, optional (default: 1e4)
            Disc radial resolution.
        """

        self.Radii=np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), steps+1)

        R_im=jscript.R_inner_mid(self)
        R_mo=jscript.R_mid_outer(self)

        if save_to_file==True:
            if path==None:
                raise ValueError('Please provide a path!')
            mypath=path

            try:
                makedirs(mypath)
            except FileExistsError:
                pass

            power=np.log10(self.Mbh/ct.MSun)

            file = open(f'{mypath}NT_inputs.txt', "w")
            file.write(f"Input Parameters:\n")
            file.write(f"version     = V1\n")
            file.write(f"log(M_SMBH) = {power}\n")
            file.write(f"spin        = {self.spin}\n")
            file.write(f"alpha       = {self.alpha}\n")
            file.write(f"mdot        = {self.mdot}\n")
            file.write(f"eps         = {self.eps}\n")
            file.write(f"le          = {self.le}\n")
            file.write(f"R_min       = {self.Rmin:.1e}\n")
            file.write(f"R_max       = {self.Rmax:.1e}\n")
            file.write(f"steps       = {steps}\n")
        
        Radiuss=[]
        sigmas=[]
        Hs=[]
        hrs=[]
        rho0s=[]
        rhos=[]
        Ts=[]
        rs=[]
        css=[]

        kappas=[]

        flag=0

        inner_flag=0
        inner_transition_flag=0
        mid_flag=0
        mid_transition_flag=0
        outer_flag=0

        k=50
        for i in range(k, steps+1):
            r=self.Radii[i]
            y=np.sqrt(r/self.Rg)
            rstar=(r/self.Rg)

            #######TO EDIT

            if rstar<0.5*R_im:
                if inner_flag==0:
                    if printing==True:
                        print(f'disk confidently in inner region')
                    inner_flag=1
                rho_0=jscript.rho_0_NT(r, self)
                T=jscript.T_NT(r, self)
                H=jscript.H_NT_2(r, self)
                sigma=jscript.Sigma_NT(r, self)
                K=0.2*(1+self.X)

            if 0.5*R_im<=rstar<5*R_im:
                if inner_transition_flag==0:
                    if printing==True:
                        print(f'disk in inner-mid transition region')
                    inner_transition_flag=1
                
                param_in=jscript.Sigma_NT(r, self)
                param_mid=jscript.Sigma_NT_Middle(r,self)

                K=0.2*(1+self.X)

                if param_in-param_mid>0:
                    if mid_flag==0:
                        if printing==True:
                            print(f'disk transitions to middle region at {rstar} Rg')
                            self.r_im=rstar
                        mid_flag=1
                    rho_0=jscript.rho_0_NT_Middle(r, self)
                    T=jscript.T_NT_Middle(r, self)
                    H=jscript.H_NT_Middle(r, self)
                    sigma=jscript.Sigma_NT_Middle(r, self)
                else:
                    sigma=jscript.Sigma_NT(r, self)
                    rho_0=jscript.rho_0_NT(r, self)
                    T=jscript.T_NT(r, self)
                    H=jscript.H_NT_2(r, self)
                
            
            if 5*R_im<=rstar<0.5*R_mo:
                rho_0=jscript.rho_0_NT_Middle(r, self)
                T=jscript.T_NT_Middle(r, self)
                H=jscript.H_NT_Middle(r, self)
                sigma=jscript.Sigma_NT_Middle(r, self)

                K=0.2*(1+self.X)

            if 0.5*R_mo<=rstar<100*R_mo:
                if mid_transition_flag==0:
                    if printing==True:
                        print(f'disk in mid-outer transition region')
                    mid_transition_flag=1
                
                param_out=jscript.Sigma_NT_Outer(r, self)
                param_mid=jscript.Sigma_NT_Middle(r, self)

                if param_mid-param_out>0:
                    if outer_flag==0:
                        if printing==True:
                            print(f'disk transitions to outer region at {rstar} Rg')
                            self.r_mo=rstar
                        outer_flag=1
                    rho_0=jscript.rho_0_NT_Outer(r, self)
                    T=jscript.T_NT_Outer(r, self)
                    H=jscript.H_NT_Outer(r, self)
                    sigma=jscript.Sigma_NT_Outer(r, self)
                    K=6.4e22 * rho_0 * T**(-7/2)
                else:
                    rho_0=jscript.rho_0_NT_Middle(r, self)
                    T=jscript.T_NT_Middle(r, self)
                    H=jscript.H_NT_Middle(r, self)
                    sigma=jscript.Sigma_NT_Middle(r, self)
                    K=0.2*(1+self.X)
            
            if 5*R_mo<rstar:
                rho_0=jscript.rho_0_NT_Outer(r, self)
                T=jscript.T_NT_Outer(r, self)
                H=jscript.H_NT_Outer(r, self)
                sigma=jscript.Sigma_NT_Outer(r, self)
                K=6.4e22 * rho_0 * T**(-7/2)

            ########END

            Cs=H/(100*np.sqrt(ct.G * self.Mbh / r**3))
            
            sigmas.append(sigma)
            Ts.append(T)
            Hs.append(H)
            rho0s.append(rho_0)
            Radiuss.append(rstar)
            rs.append(r)

            kappas.append(K)
            css.append(Cs)

            hr=H/(100*r)

            hrs.append(hr)

            rho=sigma/(2*H)
            rhos.append(rho)

            self.Omega=np.sqrt(ct.G * self.Mbh / (r*r*r))

            v=self.Omega * r
            vc=v/ct.c

            Qt= self.Omega**2 / (2 * np.pi * ct.G * rho)

            if vc<0.1 and flag==0:
                if printing==True:
                    print(f'disk stops being relativistic at {r/self.Rg} Rg')
                self.r_rel=r/self.Rg
                flag=1

        #Outputs converted to SI units
        self.Sigma=np.array(sigmas)/1e-1
        self.T=np.array(Ts)
        self.h=np.array(Hs)/1e2
        self.R=np.array(rs)
        self.rstar=np.array(Radiuss)
        self.rho=np.array(rho0s)/1e-3
        self.HRs=np.array(hrs)
        self.kappa=np.array(kappas)/1e1
        self.cs=np.array(css)

        self.tauV=self.kappa*self.rho*self.h

        self.Mdot=self.mdot * jscript.Ledd(self.Mbh, self.X)

        self.Omega = np.sqrt(ct.G * self.Mbh / (self.R*self.R*self.R))
        Teff4 = 3 * self.Mdot * (1 - np.sqrt(self.Rmin / self.R)) * self.Omega * self.Omega / (8 * np.pi)
        self.Teff4 = Teff4 / ct.sigmaSB
        
        if save_to_file==True:
            file = open(f'{mypath}NT_inputs_smooth.txt', "a")
            d=dict({'Radius [Rg]': Radiuss, 'Surface Density [gcm^-2]': sigmas, 'Temperature [K]': Ts, 'Midplane Density [gcm^-2]': rho0s, 'Thickness [cm]': Hs, 'Aspect Ratio': hrs})
            df=pd.DataFrame.from_dict(d)
            df.to_csv(f'{mypath}NT_disc_smooth.csv', index=False)

    def plot(self, params='all', path=None, save_to_file=False):
        colour='plasma'
        cmap = plt.colormaps[colour]

        power=round(np.log10(self.Mbh/ct.MSun))

        all_params = ['Sigma', 'T', 'rho0', 'H', 'HR']
        param_dic = {'Sigma': [r"$\Sigma_{\rm g} [{\rm g \, cm}^{-2}]$", self.Sigma],
                    'T': [r"$T \, [\mathrm{K}]$", self.T], 
                    'rho0': [r"$\rho \, [\mathrm{g cm^{-3}}]$", self.rho0],
                    'H': [r"$H [cm]$", self.H],
                    'HR': [r"$H/R$", self.HRs], }
        if params == "all" or set(all_params).issubset(params):
            fig, axs = plt.subplots(1, 5, figsize=(25, 5), dpi=100, tight_layout=True)
            i=0
            axs[i].plot(self.R, param_dic['Sigma'][1], '-', color=cmap(0.8), label = r"NT")
            axs[i].set_ylabel(param_dic['Sigma'][0])
            i=1
            axs[i].plot(self.R, param_dic['T'][1], '-', color=cmap(0.8), label = r"NT")
            axs[i].set_ylabel(param_dic['T'][0])
            i=2
            axs[i].plot(self.R, param_dic['rho0'][1], '-', color=cmap(0.8), label = r"NT")
            axs[i].set_ylabel(param_dic['rho0'][0])
            i=3
            axs[i].plot(self.R, param_dic['H'][1], '-', color=cmap(0.8), label = r"NT")
            axs[i].set_ylabel(param_dic['H'][0])
            i=4
            axs[i].plot(self.R, param_dic['HR'][1], '-', color=cmap(0.8), label = r"NT")
            axs[i].set_ylabel(param_dic['HR'][0])

            for ax in axs:
                ax.axvline(x=self.r_im, linestyle='--', color=cmap(0.9), alpha=0.5, label = r"$R_{inner}$")
                ax.axvline(x=self.r_rel, linestyle='--', color=cmap(0.7), alpha=0.5, label = r"$R_{rel}$")
                ax.axvline(x=self.r_mo, linestyle='--', color=cmap(0.4), alpha=0.5, label = r"$R_{outer}$")

                ax.set_xscale('log')
                ax.set_yscale('log')

                ax.set_xlabel("r/M")
                ax.set_xlim(1e0, 3e7)

            axs[i].legend()
            
        elif set(params).issubset(all_params):
            f, ax = plt.subplots(1, (params), figsize=(10, 10 + int(len(params)) * (2 / 3)), gridspec_kw=dict(hspace=0), tight_layout=True)
            ax[-1].set_xlabel(r"$r/M$")
            ax[-1].set_xscale('log')
            ax[-1].set_yscale('log')

            for i_param, param in enumerate(params):
                ax[i_param].plot(np.log10(self.R), param_dic[param][1])
                ax[i_param].set_ylabel(param_dic[param][0])
                ax[i_param].axvline(x=self.r_im, linestyle='--', color=cmap(0.9), alpha=0.5, label = r"$R_{inner}$")
                ax[i_param].axvline(x=self.r_rel/self.Rg, linestyle='--', color=cmap(0.7), alpha=0.5, label = r"$R_{rel}$")
                ax[i_param].axvline(x=self.r_mo, linestyle='--', color=cmap(0.4), alpha=0.5, label = r"$R_{outer}$")
            ax[-1].legend()

        plt.suptitle(f'$SMBH = 10^{power}'r'{M_{\odot}}, \alpha$ = 'f'{self.alpha},'r'$\chi$ = 'f'{self.spin}')
        
        if save_to_file==True:
            if path==None:
                raise ValueError('Please provide a path!')
            mypath=path
            plt.savefig(f'{mypath}all_profiles_with_TQM_smooth.pdf')
        
        plt.show()
