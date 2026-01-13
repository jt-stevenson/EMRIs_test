
import matplotlib.pyplot as plt
import pagn.constants as ct
import multiprocessing
import numpy as np
import argparse
import warnings
import pagn
import time
import os
from datetime import datetime
from datetime import datetime
from tqdm import tqdm
from scipy.integrate import solve_ivp

start = datetime.now()
warnings.filterwarnings('ignore')

printing=False
plotting=True
type_II_computation = "conservative" 
C=965665
Spin=0.9

Fixed=True

winds=True

import binary_formation_distribution_V8 as myscript
import binary_formation_distribution_V10 as myscript2
import NT_disk_Eqns_V1 as jscript
import Novikov

################################################################################################
### Simulation routine #########################################################################
################################################################################################xs
def iteration(args, MBH, T, mass_sec, mass_prim_vk):
    # initialize random number generator for the radii and the masses
    seed = (os.getpid() + int(time.time() * 1e6)) % 2**32
    np.random.seed(seed)

    # Initialization of the Disk 

    if Fixed==False:
        c = np.random.randint(0, len(MBH))
        spin=np.random.rand()
    if Fixed==True:
        c=args.c
        spin=args.spin
    
    Mbh = MBH[c] * ct.MSun    # M_SMBH
    T = T[c] * 1e6 * ct.yr           # disk lifetime
    alpha = args.a                   # viscosity parameter

    if args.DT  == "SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha)
        Rmin = disk.Rmin
        Rmax = disk.Rmax
    elif args.DT  == "TQM":
        # R_G=MBH * ct.G /(ct.c*ct.c)
        # Rout=1e7*R_G
        # sigma = (200 * 1e3) * (MBH / (1.3e8*ct.MSun)) ** (1 / 4.24)
        # Mdot_out = 320*(ct.MSun/ct.yr)*(Rout/(95*ct.pc)) * (sigma/(188e3))**2
        disk = pagn.ThompsonAGN(Mbh=Mbh, Mdot_out=None)
        Rmin = disk.Rin
        Rmax = disk.Rout
        # print(disk.Mdot_out)
    elif args.DT  == "NT":
        disk = Novikov.NovikovThorneAGN(Mbh=Mbh, alpha=alpha, spin=spin)
        Rmin = disk.Rmin
        Rmax = disk.Rmax
    disk.solve_disk()

    Rsch = 2 * ct.G * Mbh / ct.c**2
    Rs = disk.R / Rsch

    # Mmean=np.mean(mass_sec) * ct.MSun
    Mmean=np.mean(mass_sec)* ct.MSun 

    ledd=jscript.Ledd(Mmean, X=0.7)
    Ledd=jscript.Ledd(Mbh, X=0.7)

    le=1
    eps=0.1

    mdot=le * ledd / (eps * ct.c*ct.c)
    Mdot=le * Ledd / (eps * ct.c*ct.c)
    
    mean_Gamma_GW = 1e-7 * jscript.compute_torque_GW(args, disk, Mmean, Mbh) 
    mean_Gamma_noGW = 1e-7 * jscript.compute_noGW_torque(args, disk, Mmean, Mbh) 

    if winds==True:
        mbhl=jscript.BHL_accretion(args, disk, Mbh, Mmean, Mdot)
        # print(mbhl)

        drhodr=jscript.drhodR(disk)

        mean_Gamma_wind= jscript.compute_torque_wind(disk, mbhl, Mmean) 
        #mean_Gamma_wind=10**7*myscript2.gamma_wind(Mmean, disk, drhodr)
        mean_Gamma =  (mean_Gamma_GW+mean_Gamma_noGW+mean_Gamma_wind)
    else:
        mean_Gamma =  (mean_Gamma_GW+mean_Gamma_noGW)

    traps = myscript.mig_trap(disk, mean_Gamma_noGW) 
    traps=np.array(traps)/Rsch

    fig, ax = plt.subplots(figsize=(8, 6))

    for trap in traps:
        ax.axvline(trap *2, color='gray', linestyle=':', linewidth=1, label='Migration trap' if trap == traps[0] else "")

    # Mean line: solid for Gamma > 0, dashed for Gamma < 0
    # Split into continuous regions of positive or negative torque
    
    sign_changes2 = np.where(np.diff(np.sign(mean_Gamma_GW)) != 0)[0]
    split_indices2 = np.concatenate(([0], sign_changes2 + 1, [len(Rs)]))

    for i in range(len(split_indices2) - 1):
        start, end = split_indices2[i], split_indices2[i + 1]
        R_seg = Rs[start:end] *2
        G_seg = mean_Gamma_GW[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='$|\Gamma_{GW}| >0$' if i == 1 else "", color='hotpink')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='$|\Gamma_{GW}| <0$' if i == 0 else "", color='hotpink')

    sign_changes3 = np.where(np.diff(np.sign(mean_Gamma_noGW)) != 0)[0]
    split_indices3 = np.concatenate(([0], sign_changes3 + 1, [len(Rs)]))

    for i in range(len(split_indices3) - 1):
        start, end = split_indices3[i], split_indices3[i + 1]
        R_seg = Rs[start:end] *2
        G_seg = mean_Gamma_noGW[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='$|\Gamma_{typeI}| >0$' if i == 1 else "", color='royalblue')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='$|\Gamma_{typeI}| <0$' if i == 0 else "", color='royalblue')
    if winds==True:
        sign_changes4 = np.where(np.diff(np.sign(mean_Gamma_wind)) != 0)[0]
        split_indices4 = np.concatenate(([0], sign_changes4 + 1, [len(Rs)]))

        for i in range(len(split_indices4) - 1):
            start, end = split_indices4[i], split_indices4[i + 1]
            R_seg = Rs[start:end] *2
            G_seg = mean_Gamma_wind[start:end]
            if np.all(G_seg > 0):
                ax.plot(R_seg, np.abs(G_seg), 'k-', label='$|\Gamma_{wind}| >0$' if i == 1 else "", color='mediumorchid')
            elif np.all(G_seg < 0):
                ax.plot(R_seg, np.abs(G_seg), 'k--', label='$|\Gamma_{wind}| <0$' if i == 0 else "", color='mediumorchid')

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

    MBH_power=np.log10(Mbh/ct.MSun)
    power=round(np.log10(Mbh/ct.MSun),2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'R [R$_{\rm g}$]')
    ax.set_ylabel(r'$|\Gamma|$ [kg]')
    ax.set_ylim([1e22,1e35])
    ax.set_title(f'Migration Torques ($SMBH = 10^{power}$'r'${M_{\odot}})$')
    ax.legend()
    plt.tight_layout()
    if args.DT  == "TQM":
        plt.savefig(f'Torques/TQM/Mbh_{np.log10(Mbh/ct.MSun):.1f}_{args.DT}_{args.TT}_{args.gen}_wind_MP.pdf', format='pdf', dpi=300)
    elif args.DT == "SG":
        plt.savefig(f'Torques/SG/Mbh_{np.log10(Mbh/ct.MSun):.2f}_alpha_{args.a}_{args.DT}_{args.TT}_{args.gen}_MP_2.pdf', format='pdf', dpi=300)
    elif args.DT == "NT":
        plt.savefig(f'Torques/NT/Mbh_{np.log10(Mbh/ct.MSun):.1f}_alpha_{args.a}_spin_{args.spin}_{args.DT}_{args.TT}_{args.gen}.pdf', format='pdf', dpi=300)
    return

################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM', 'NT'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-a', type=float, default=0.1)    # real number
    parser.add_argument('-spin', type=float, default=0.9)
    parser.add_argument('-N', type=int, default=5000) # integer number
    parser.add_argument('-c', type=int, default=965665)
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
    mass_sec=np.genfromtxt("BHs_single_Zsun_rapid_nospin.dat",usecols=(0),skip_header=3,unpack=True)
    mass_prim_vk = np.genfromtxt('Ng_catalog.txt', skip_header=1)
    MBH, T = np.genfromtxt("SMBHmass_local_AGNlifetime_pairs.txt", unpack=True, skip_header=3)
    iteration(args, MBH, T, mass_sec, mass_prim_vk)
################################################################################################

################################################################################################
### Output file ################################################################################
################################################################################################
    if printing == True:
        date_time = start.strftime("%y%m%d_%H%M%S")

        dir_name = f"EMRIs_Jupiter_2/c_{C}/{args.DT}/alpha_{args.a}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name+f"/EMRIs_{args.TT}_{args.gen}_{args.N}_events_with_GW.txt"
        file_name_1g = dir_name+f"/EMRIs_{args.TT}_1g_{args.N}_events_with_GW.txt"
        if args.gen=='Ng' and  not os.path.exists(file_name_1g):
            print()
            print('There is no 1g source for this Ng simulation. Run the same simulation for 1g first!')
            quit()
        file = open(file_name, "w")

        # print all parameters in the file
        file.write(f"Parameters:\n")
        file.write(f"version = V1\n")
        file.write(f"date_time = {date_time}\n")
        file.write(f"comp_time = {0}\n")
        file.write(f"disk_type = {args.DT}\n")
        file.write(f"torque_type = {args.TT}\n")
        file.write(f"alpha = {args.a:.3f}\n")
        file.write(f"gen = {args.gen}\n")
        file.write(f"N = {args.N}\n")
        if Fixed==True:
            file.write(f'M_smbh = {np.log10(MBH[C]):.3f}\n')
            file.write(f'Spin = {Spin}\n')
            file.write(f'T = {T[C]/(1e6*ct.yr):.3e}\n')
        file.write(f"\n")
        file.write(f"Data:\n")
        if Fixed==True:
            file.write(f"m1/Msun, r0/Rg, t_gw/Myr, t_migr/Myr, is_emri, Ng, R_final/Rg, lisa_radii/Rg, t_lisa/Myr, t_final/Myr, lisa_flag, total_flags\n")
        elif Fixed==False:
            file.write(f"logMBH/Msun, m1/Msun, r0/Rg, chi_eff, T/Myr, t_gw/Myr, t_migr/Myr, is_emri, Ng, R_final/Rg, lisa_radii/Rg, lisa_exit_radii/Rg, t_lisa/Myr, t_final/Myr, lisa_flag, total_flags\n")
################################################################################################

################################################################################################
### Loading of 1g r_pu data ####################################################################
################################################################################################
        if args.gen=='1g':
            r_pu_1g = np.array([])
        if args.gen=='Ng':
            params, data = myscript.load_file(file_name_1g)
            r_pu_1g = data['r_pu'][data['paired']==1]
################################################################################################

################################################################################################
### Determination of binary formation radii ####################################################
################################################################################################
        print()
        N_batches = 10
        N_iter = int(args.N)
        chunk_size = int(N_iter/N_batches)

        # run parallel simulations and print results in file
        #with multiprocessing.Pool(1) as pool:
        with multiprocessing.Pool(os.cpu_count()) as pool:
            with tqdm(total=N_iter) as pbar:
                for i in range(N_batches):
                    input_data = [(args, MBH, T, mass_sec, mass_prim_vk, r_pu_1g) for _ in range(chunk_size)]
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


