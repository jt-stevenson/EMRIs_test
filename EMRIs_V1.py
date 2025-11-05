import binary_formation_distribution_V8 as myscript
import NT_disk_Eqns_V1 as jscript

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

start = datetime.now()
warnings.filterwarnings('ignore')

printing=True
plotting=True

################################################################################################
### Simulation routine #########################################################################
################################################################################################
def iteration(args, MBH, T, mass_sec, mass_prim_vk, r_pu_1g):
    # initialize random number generator for the radii and the masses
    seed = (os.getpid() + int(time.time() * 1e6)) % 2**32
    np.random.seed(seed)

    # Initialization of the Disk 
    c = np.random.randint(0, len(MBH))
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
        print(disk.Mdot_out)
    disk.solve_disk()

    # migrator mass
    a = np.random.randint(0,len(mass_sec),1)
    m1 = mass_sec[a]*ct.MSun
    if args.gen=='Ng':
        mass_prim = mass_prim_vk[:, 0]
        a = np.random.randint(0,len(mass_prim))
        m1 = mass_prim[a]*ct.MSun
        Ng = mass_prim_vk[a, 2]
    elif args.gen=='1g':
        Ng = 1
    m1=float(m1)


    # compute torques
    Gamma = myscript.compute_torque_function(args, disk, m1, Mbh) 

    # compute trap locations 
    traps = myscript.mig_trap(disk, Gamma(disk.R)) 
    innermost_trap = traps[0] if len(traps) > 0 else Rmax
    
    # time and initial radius
    t0 = 0
    r0 = 10**(np.log10(Rmin) + (np.log10(innermost_trap) - np.log10(Rmin)) * np.random.rand())
    Rs = 2*ct.G*Mbh/ct.c**2
    if args.gen=='Ng':
        b = np.random.randint(0, len(r_pu_1g))
        r1 = r_pu_1g[b]
        r0 = myscript.pos_after_kick(r1, mass_prim_vk, Mbh)
    
    # select as EMRI if inside innermost trap, migrating inward
    Gamma_r0 = Gamma(r0)
    emri_flag = (Gamma_r0 < 0) ###and (r0 < innermost_trap) #NB: this condition is now enforced in extraction of r0


    # Migration timescale = L / |Gamma|
    L = m1 * np.sqrt(ct.G * Mbh * r0)
    t_migr = L / np.abs(Gamma_r0)

    # GW inspiral time 
    t_gw = (5 / 256) * (ct.c**5 / (ct.G**3)) * (r0**4) / (m1 * Mbh**2)

    # ispiral happens within disk's lifetime
    emri_within_T = min(t_migr, t_gw) < T

    # for EMRIs this is not a binary quantity but depends only on SMBH spin (assumed random)
    spin=np.random.rand()
    # future task - add SMBH spin distribution to draw from?
    chi_eff = 2 * spin - 1  # in [-1, 1]

    # final flag
    is_emri = emri_flag and emri_within_T

    # code frankesteined in by Jupiter

    R_isco=jscript.R_isco_function(MBH, spin)

    if is_emri is True:
        Lisa_flag, Lisa_radii=jscript.LISAband_flag(r0, R_isco, MBH, m, print = True)
        # print(Lisa_radii)

    
    #assume zero eccentricity
    return f"{np.log10(Mbh/ct.MSun):.1f} {m1/ct.MSun:.3e} {r0/Rs:.3e} {chi_eff:.3e} {T/(1e6*ct.yr):.3e} {t_gw/(1e6*ct.yr):.3e} {t_migr/(1e6*ct.yr):.3e} {is_emri} {Ng}\n"


################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-a', type=float, default=0.01)    # real number
    parser.add_argument('-N', type=int, default=100) # integer number
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
################################################################################################



################################################################################################
### Output file ################################################################################
################################################################################################
    if printing == True:
        date_time = start.strftime("%y%m%d_%H%M%S")

        dir_name = f"EMRIs/{args.DT}/alpha_{args.a}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name+f"/EMRIs_{args.TT}_{args.gen}.txt"
        file_name_1g = dir_name+f"/EMRIs_{args.TT}_1g.txt"
        if args.gen=='Ng' and  not os.path.exists(file_name_1g):
            print()
            print('There is no 1g source for this Ng simulation. Run the same simulation for 1g first!')
            quit()
        file = open(file_name, "w")

        # print all parameters in the file
        file.write(f"Parameters:\n")
        file.write(f"version     = V1\n")
        file.write(f"date_time   = {date_time}\n")
        file.write(f"comp_time   = {0}\n")
        file.write(f"disk_type   = {args.DT}\n")
        file.write(f"torque_type = {args.TT}\n")
        file.write(f"alpha       = {args.a:.3f}\n")
        file.write(f"gen         = {args.gen}\n")
        file.write(f"N           = {args.N}\n")
        file.write(f"\n")
        file.write(f"Data:\n")
        file.write(f"logMBH/Msun, m1/Msun, r0/Rs, chi_eff, T/Myr, t_gw/Myr, t_migr/Myr, is_emri, Ng \n")
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