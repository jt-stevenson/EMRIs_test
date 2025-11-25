
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
C=392643
Spin=0.9

Fixed=True

import binary_formation_distribution_V8 as myscript
import NT_disk_Eqns_V1 as jscript

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
        c=C
        spin=Spin
    
    Mbh = MBH[c] * ct.MSun    # M_SMBH
    T = T[c] * 1e6 * ct.yr           # disk lifetime
    alpha = args.a                   # viscosity parameter

    if args.DT  == "SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha)
        Rmin = disk.Rmin
        Rmax = disk.Rmax
    elif args.DT  == "TQM":
        disk = pagn.ThompsonAGN(Mbh=Mbh, Mdot_out=None)
        Rmin = disk.Rin
        Rmax = disk.Rout
        # print(disk.Mdot_out)
    disk.solve_disk()

    Rsch = 2 * ct.G * Mbh / ct.c**2
    Rs = disk.R / Rsch
    # Gammas = []
    # Gammas_GW=[]
    # Gammas_noGW=[]

    Mmean=np.mean(mass_sec) * ct.MSun

    # for M in mass_sec:
    #     m1=mass_sec*ct.MSun
    #     print(args, disk, m1, Mbh)
    #     Gamma = myscript.compute_torque_function(args, disk, m1, Mbh)
    #     Gamma_GW = jscript.compute_GW_torque_function(args, disk, m1, Mbh)
    #     Gamma_no_GW=jscript.compute_noGW_torque_function(args, disk, m1, Mbh)

    #     Gammas.append(Gamma)
    #     Gammas_GW.append(Gamma_GW)
    #     Gammas_noGW.append(Gamma_no_GW)

    # Gammas = np.array(Gammas)
    # Gammas_GW=np.array(Gammas_GW)
    # Gammas_noGW=np.array(Gammas_noGW)

    # mean_Gamma = myscript.compute_torque(args, disk, Mmean, Mbh)
    mean_Gamma_GW = jscript.compute_torque_GW(args, disk, Mmean, Mbh)
    mean_Gamma_noGW=jscript.compute_noGW_torque(args, disk, Mmean, Mbh)
    mean_Gamma = mean_Gamma_GW+mean_Gamma_noGW

    traps = myscript.mig_trap(disk, mean_Gamma_noGW) 
    traps=np.array(traps)/Rsch
    # min_Gamma = np.min(Gammas, axis=0)
    # max_Gamma = np.max(Gammas, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # ax.fill_between(Rs, np.abs(min_Gamma), np.abs(max_Gamma), alpha=0.3, label='Migrator mass range')
    for trap in traps:
        ax.axvline(trap, color='gray', linestyle=':', linewidth=1, label='Migration trap' if trap == traps[0] else "")

    # Mean line: solid for Gamma > 0, dashed for Gamma < 0
    # Split into continuous regions of positive or negative torque
    
    sign_changes2 = np.where(np.diff(np.sign(mean_Gamma_GW)) != 0)[0]
    split_indices2 = np.concatenate(([0], sign_changes2 + 1, [len(Rs)]))

    for i in range(len(split_indices2) - 1):
        start, end = split_indices2[i], split_indices2[i + 1]
        R_seg = Rs[start:end]
        G_seg = mean_Gamma_GW[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='|Gamma_GW| >0' if i == 1 else "", color='hotpink')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='|Gamma_GW| <0' if i == 0 else "", color='hotpink')

    sign_changes3 = np.where(np.diff(np.sign(mean_Gamma_noGW)) != 0)[0]
    split_indices3 = np.concatenate(([0], sign_changes3 + 1, [len(Rs)]))

    for i in range(len(split_indices3) - 1):
        start, end = split_indices3[i], split_indices3[i + 1]
        R_seg = Rs[start:end]
        G_seg = mean_Gamma_noGW[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='|Gamma_noGW| >0' if i == 1 else "", color='royalblue')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='|Gamma_noGW| <0' if i == 0 else "", color='royalblue')

    sign_changes = np.where(np.diff(np.sign(mean_Gamma)) != 0)[0]
    split_indices = np.concatenate(([0], sign_changes + 1, [len(Rs)]))

    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        R_seg = Rs[start:end]
        G_seg = mean_Gamma[start:end]
        if np.all(G_seg > 0):
            ax.plot(R_seg, np.abs(G_seg), 'k-', label='|Gamma| >0' if i == 1 else "", color='rebeccapurple')
        elif np.all(G_seg < 0):
            ax.plot(R_seg, np.abs(G_seg), 'k--', label='|Gamma| <0' if i == 0 else "", color='rebeccapurple')

    MBH_power=np.log10(Mbh/ct.MSun)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'R [R$_{\rm S}$]')
    ax.set_ylabel(r'$|\Gamma|$ [cgs]')
    ax.set_ylim([1e29,1e44])
    ax.set_title(f'Migration Torques ($SMBH = 10^{MBH_power:.0f}$'r'${M_{\odot}})$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'Torques/Mbh_{np.log10(Mbh/ct.MSun):.1f}_alpha_{args.a}_{args.DT}_{args.TT}_{args.gen}.png', format='png', dpi=300)
    return

################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-a', type=float, default=0.1)    # real number
    parser.add_argument('-N', type=int, default=5000) # integer number
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


