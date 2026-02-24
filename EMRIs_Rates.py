
import matplotlib.pyplot as plt
import pagn.constants as ct
import multiprocessing
import numpy as np
import pandas as pd
import argparse
import warnings
import pagn
import time
import os
from datetime import datetime
from datetime import datetime
from tqdm import tqdm
from scipy.integrate import solve_ivp

import powerlaw
from scipy.interpolate import interp1d

start = datetime.now()
warnings.filterwarnings('ignore')

printing=True
plotting=True
type_II_computation = "conservative" 
Fixed=True

import binary_formation_distribution_V11 as myscript #edited to explicitly take alpha instead of disc.alpha
import NT_disk_Eqns_V2 as jscript
import Novikov

################################################################################################
### Simulation routine #########################################################################
################################################################################################xs
def iteration(args, cluster_df, mass_prim_vk, r_pu_1g, disk, N):
    # initialize random number generator for the radii and the masses
    seed = (os.getpid() + int(time.time() * 1e6)) % 2**32
    np.random.seed(seed)
    
    spin=args.spin
    
    Mbh = args.Mbh * ct.MSun    # M_SMBH
    T = args.T * 1e6 * ct.yr # disk lifetime
    alpha = args.a # viscosity parameter

    R_g=Mbh * ct.G /(ct.c*ct.c)

    Rmax=disk.Rmax
    Rmin=disk.Rmin

    # migrator mass
    a = np.random.randint(0, N ,1)

    m1 = cluster_df['mbh [Msun]'][a]*ct.MSun

    if args.gen=='Ng':
        mass_prim = mass_prim_vk[:, 0]
        a = np.random.randint(0,len(mass_prim))
        m1 = mass_prim[a]*ct.MSun
        Ng = mass_prim_vk[a, 2]
    elif args.gen=='1g':
        Ng = 1
    m1=float(m1)

    # compute torques
    # print(f'm1/Mbh: {m1/Mbh}')
    Gamma = myscript.compute_torque_function(args, disk, m1, Mbh) 

    # compute trap locations 
    traps = myscript.mig_trap(disk, Gamma(disk.R)) 
    innermost_trap = traps[0] if len(traps) > 0 else Rmax
    
    # time and initial radius
    t0 = 0
    r0=cluster_df['r [Rg]'][a]*R_g
    cos_i=cluster_df[cos_i][a]

    if args.gen=='Ng':
        b = np.random.randint(0, len(r_pu_1g))
        r1 = r_pu_1g[b]
        r0 = myscript.pos_after_kick(r1, mass_prim_vk, Mbh)
    
    # select as EMRI if inside innermost trap, migrating inward
    Gamma_r0 = Gamma(r0)
    emri_flag = (Gamma_r0 < 0) and (r0 < innermost_trap) #NB: this condition is no longer enforced in extraction of r0

    # Alignment timescale, Rowan et al 2024
    f=interp1d(disk.R, disk.h, kind='linear', fill_value='extrapolate')
    H=f(r0)
    t_align = jscript.T_align(disk, Mbh, m1, cos_i, H, r0)

    # Encounter timescale, Rowan et al 2024
    t_enc = jscript.T_enc(Mbh, m1, r0, N, disk)

    # Migration timescale = L / |Gamma|
    L = m1 * np.sqrt(ct.G * Mbh * r0)
    t_migr = L / np.abs(Gamma_r0)

    # GW inspiral time
    t_gw = (5 / 256) * (ct.c**5 / (ct.G**3)) * (r0**4) / (m1 * Mbh**2)

    # alignment and inspiral happen within disk's lifetime + sbh not scattered faster than alignment
    align = t_align < T and t_align < t_enc
    emri_within_T = min(t_migr, t_gw) < T and align

    # final flag
    is_emri = emri_flag and emri_within_T

    if emri_flag==True:
        inspiral_flag=1
    if emri_flag==False:
        inspiral_flag=0

    if emri_within_T==True:
        Tdisc_flag=2
    if emri_within_T==False:
        Tdisc_flag=0

    if is_emri==True:
        is_EMRI=2
    elif is_emri==False:
        is_EMRI=0

    # code frankesteined in from binary_formation_distribution_V8 by Jupiter, Some Original Code

    lisa_entry_radii=0
    lisa_exit_radii=0
    lisa_flag=0

    t_lisa_entry=0
    t_lisa_exit=0

    t_inspiral=0

    M1=np.array([m1])

    gap = np.full(M1.shape, 'type_I', dtype='<U7')

    Gammas = myscript.compute_torque_function(args, disk, m1, Mbh)
    gap_event_1 = myscript.type_II_event(disk, m1, Mbh)
    R_event_1=jscript.r_isco_event(Mbh, m1, spin)

    # print(f'SMBH {Mbh/ct.MSun:.3e} Msun and SBH {m1/ct.MSun:.3e} Msun, r0 {r0/R_g:.3e} Rg')

    GWf=jscript.GW_freq_fn(r0, Mbh, m1)
    if 1.0>GWf>0.0001:
        lisa_radii=r0
        lisa_flag=4
        

    solution1 = solve_ivp(fun=myscript.rdot, 
                            t_span=[t0, T], 
                            y0=[r0],
                            method='RK23', 
                            events=(myscript.trap_dist, gap_event_1, jscript.LISA_band_enter , R_event_1), #, Rs_event_1)
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas, Mbh, traps))

    t1 = solution1.t
    r1 = solution1.y[0]

    #check whether "events" happened (reaching Type I migration trap or doing Type II migration, or crossing inner edge of disk or if LISA band entered/exited)

    # if you open a gap
    # keep integrating with Type II torque
    if len(solution1.t_events[1])>0:
        print("SBH does type II")
        gap[0]='type_II'
        t_gap = solution1.t_events[1][0]
        r_gap = solution1.y_events[1][0][0]
        # Stage 2 : Type II migration
        if (type_II_computation == "conservative") or (type_II_computation == "conservative+extra_h_condition"):
            solution1_TypeII = solve_ivp(myscript.rdot_typeII, [t_gap, T], [r_gap],
                                     args=(disk, alpha),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        elif type_II_computation == "modified_type_I":
            solution1_TypeII = solve_ivp(myscript.rdot_typeII_Kanagawa2018, [t_gap, T], [r_gap],
                                     args=(M[0], disk, Mbh, alpha),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        print("integration type II concluded")
        # Stitch the two segments 
        t1 = np.concatenate((t1, solution1_TypeII.t[1:]))     # skip duplicate t_gap
        r1 = np.concatenate((r1, solution1_TypeII.y[0][1:]))
    
    elif len(solution1.t_events[0])>0:  #checks if trap is reached- shouldn't happen due to conditions on r0, but want to keep incase we begin integrating from further out than inside the first trap
        t1 = np.append(t1, T)           #or for some other weirdness
        r1 = np.append(r1, r1[-1])
    
    else: 
        if len(solution1.t_events[2])>0: #check if sbh has entered the LISA band
            print('SBH has entered LISA band')
            lisa_entry_radii=solution1.y_events[2][0][0]
            t_lisa_entry=solution1.t_events[2][0]
            lisa_flag=4
            solution1_lisa_entry = solve_ivp(fun=myscript.rdot, 
                            t_span=[t_lisa_entry, T], 
                            y0=[lisa_entry_radii],
                            method='RK23', 
                            events=(jscript.LISA_band_exit, R_event_1),
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas, Mbh, traps))
            t2=solution1_lisa_entry.t[1:]
            r2=solution1_lisa_entry.y[0][1:]

            if len(solution1_lisa_entry.t_events[0])>0: #check if sbh leaves the lisa band
                print('SBH has left LISA band')
                lisa_exit_radii = solution1_lisa_entry.y_events[0][0][0]
                t_lisa_exit=solution1_lisa_entry.t_events[0][0]
                lisa_flag=4
                solution1_lisa_exit = solve_ivp(fun=myscript.rdot, 
                            t_span=[t_lisa_exit, T], 
                            y0=[lisa_entry_radii],
                            method='RK23', 
                            events=(R_event_1), 
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas, Mbh, traps))
                t3=solution1_lisa_exit.t[1:]
                r3=solution1_lisa_exit.y[0][1:]

                if len(solution1_lisa_exit.t_events[0])>0: #check if sbh crosses SMBH event horizon after leaving LISA band
                    print('SBH has crossed EH')
                    t_inspiral=t3[len(t3)-1]

                    t4 = np.append(t3, T)
                    r4 = np.append(r3, r3[-1])

                    t3=np.concatenate((t3, t4))
                    r3=np.concatenate((r3, r4))

                t2 = np.concatenate((t2, t3))
                r2 = np.concatenate((r2, r3))

            elif len(solution1_lisa_entry.t_events[1])>0 and len(solution1_lisa_entry.t_events[0])==0: #check if sbh crosses SMBH event horizon if LISA band not left
                print('SBH has crossed EH')
                t_inspiral=t2[len(t2)-1]

                t2 = np.append(t2, T)
                r2 = np.append(r2, r2[-1])

                # t2 = np.concatenate((t2, t3))
                # r2 = np.concatenate((r2, r3))

            t1 = np.concatenate((t1, t2))
            r1 = np.concatenate((r1, r2))
        
        elif len(solution1.t_events[2])==0 and len(solution1.t_events[3])>0: #check if sbh crosses SMBH event horizon if lisa band never entered
            print('SBH has crossed EH')
            t_inspiral=t1[len(t1)-1]

            t1 = np.append(t1, T)
            r1 = np.append(r1, r1[-1])

    #Jupiter Original Code... DRIED EMRI INVESTIGATION REMOVED FOR CLARITY

    #We want to find the location of the secondary (SBH) when the disc disperses to check if the signal produced is in the LISA band
    t_final=t1[len(t1)-1]
    r_final=r1[len(r1)-1]

    if t_inspiral==0: #to edit
        t_gw = (5 / 256) * (ct.c**5 / (ct.G**3)) * (r_final**4) / (m1 * Mbh**2)
        t_inspiral=t_final+t_gw

    

    rG=ct.G*Mbh*(1/(ct.c*ct.c))

    if t_final!=T and printing==True:
        print(f'SOMETHING HAS GONE WRONG! T={T/(1e6*ct.yr):.3e}!=t_final={t_final/(1e6*ct.yr):.3e}')
    
    if printing==True:
        print(f' For SMBH {Mbh/ct.MSun:.3e} Msun and SBH {m1/ct.MSun:.3e} Msun with r0 {r0/rG:.3e} Rg (Rmin: {Rmin/R_g:.3e} Rg, Rmax: {Rmax/R_g:.3e} Rg) \nAt time T={T/(1e6*ct.yr):.3e}={t_final/(1e6*ct.yr):.3e} Myrs, R_final={r_final/rG:.3e} Rg\nemri flag {emri_flag}, emri within T {emri_within_T}')
    M=Mbh+m1

    # lisa_flag, lisa_radii=jscript.LISAband_flag(r0, r_final, Mbh, m1)
    lisa_radii=lisa_entry_radii

    if t_lisa_exit==0:
        t_lisa=t_lisa_entry-t_final
    elif t_lisa_exit!=0:
        t_lisa=t_lisa_entry-t_lisa_exit

    r_isco=jscript.R_isco_function(Mbh, spin)

    if r_final<r_isco:
        r_final=r_isco

    # if lisa_flag!=0:
    #     print(f'EMRI with SMBH {MBH/ct.MSun:.3e} MSun, SBH {m1/ct.MSun:.3e} MSun, SMBH spin {spin:.3e} enters LISA band at {lisa_radii/rG:.3e} R_G')
    # elif lisa_flag==0:
    #     print(f'EMRI doesnt enter LISA band')

    # Flag to indicate one of four outcomes for plotting - INCORRECT, NEEDS UPDATING
    # final_flag=0, no inspiral w/in Tdisc, undetectable by LISA
    # final_flag=1, detectable by LISA, does not inspiral within Tdisc
    # final_flag=2, inspiral w/in T_disc, undetectable by LISA
    # final_flag=4, inspirals w/in T_disc, detectable by LISA
    final_flag=is_EMRI+lisa_flag

    t_lisa=np.abs(t_lisa)

    #flags=0, no inspiral, no signif migration within tdisk, undetected by LISA
    #flags=1, inspiral, no signif migration within tdisk, undetected by LISA
    #flags=2, no inspiral, signif migration within tdisk, undetected by LISA
    #flags=3, inspiral, signif migration within tdisk, undetected by LISA
    #flags=4, no inspiral, no signif migration within tdisk, detected by LISA
    #flags=5, inspiral, no signif migration within tdisk, detected by LISA
    #flags=6, no inspiral, signif migration within tdisk, detected by LISA
    #flags=7, inspiral, signif migration within tdisk, detected by LISA
    flags=inspiral_flag+Tdisc_flag+lisa_flag

    #assume zero eccentricity
    return f"{m1/ct.MSun:.3e} {r0/rG:.3e} {cos_i} {t_align/(1e6*ct.yr):.3e} {t_gw/(1e6*ct.yr):.3e} {t_migr/(1e6*ct.yr):.3e} {t_inspiral/(1e6*ct.yr):.3e} {is_emri} {Ng} {r_final/rG:.3e} {lisa_radii/rG:.3e} {t_lisa/(1e6*ct.yr):.3e} {t_final/(1e6*ct.yr):.3e} {lisa_flag} {flags}\n"    
################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM', 'NT'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-BIMF', type=str, default="Vaccaro", choices=['Vaccaro', 'Tagawa', 'NT'])
    parser.add_argument('-a', type=float, default=0.1)
    parser.add_argument('-le', type=float, default=0.01)
    parser.add_argument('-spin', type=float, default=0.9)    # real number
    parser.add_argument('-Mbh', type=float, default=1e6)   # MSun
    parser.add_argument('-T', type=float, default=1e7)     # Myrs
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

    Mbh=args.Mbh*ct.MSun
    alpha=args.a
    spin=args.spin
    le=args.le

    try:
        cluster_df=pd.read_csv('fEMRI_Rates/{args.BIMF}/dataframes/{args.DT}_{Mbh}_alpha_{args.alpha}_eps_{args.eps}_le_{args.le}_spin_{args.spin}.csv')
    except FileNotFoundError:
        cluster_df=jscript.cluster_sampling(Mbh, args.alpha, args.spin, args.le, args.DT, args.BIMF)
    N=len(cluster_df)

    if args.DT  == "SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha, le=le)
        Rmax = disk.Rmaxs
    elif args.DT  == "TQM":
        Rout=1e7 * 2 * ct.G * Mbh/ct.c**2
        sigma = (200 * 1e3) * (Mbh / (1.3e8*ct.MSun)) ** (1 / 4.24)
        Mdot_out = 320*(ct.MSun/ct.yr)*(Rout/(95*ct.pc)) * (sigma/(188e3))**2
        disk = pagn.ThompsonAGN(Mbh=Mbh, Mdot_out=Mdot_out)
        Rmax = disk.Rout

        ledd=jscript.Ledd(Mbh, X=0.7)
        Mdot= ledd /(ct.c**2 * 0.1)
        alpha=Mdot/(6*np.pi * disk.h * disk.h * disk.rho * disk.cs)
        alpha=np.mean(alpha)
        print(f'alpha: {alpha}')
        # print(disk.Mdot_out)
    elif args.DT  == "NT":
        disk = Novikov.NovikovThorneAGN(Mbh=Mbh, alpha=alpha, spin=spin)
        Rmax = disk.Rmax
    disk.solve_disk()

    # if args.BIMF=='Vaccaro':
    #     mass_sec=np.genfromtxt("/Users/pmxks13/PhD/EMRIs_test/BHs_single_Zsun_rapid_nospin.dat",usecols=(0),skip_header=3,unpack=True)
    # elif args.BIMF=='Bartos':
    #     mass_sec=np.genfromtxt("/Users/pmxks13/PhD/EMRIs_test/BHs_Bartos_exp_2.dat",usecols=(0),skip_header=3,unpack=True)
    # elif args.BIMF=='Tagawa':
    #     mass_sec=np.genfromtxt("/Users/pmxks13/PhD/EMRIs_test/BHs_Tagawa_exp_2.3.dat",usecols=(0),skip_header=3,unpack=True)


    mass_prim_vk = np.genfromtxt('/Users/pmxks13/PhD/EMRIs_test/Ng_catalog.txt', skip_header=1)


    # MBH, T = np.genfromtxt("/Users/pmxks13/PhD/EMRIs_test/SMBHmass_local_AGNlifetime_pairs.txt", unpack=True, skip_header=3)
################################################################################################

################################################################################################
### Output file ################################################################################
################################################################################################
    if printing == True:
        date_time = start.strftime("%y%m%d_%H%M%S")

        print('printing to file...')

        dir_name = f"/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/{args.BIMF}/Mbh_{args.Mbh:.1e}/{args.DT}/alpha_{args.a}/spin_{args.spin}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name+f"/EMRIs_{args.TT}_{args.gen}_{N}.txt"
        print(f'file name: {file_name}')
        file_name_1g = dir_name+f"/EMRIs_{args.TT}_1g_{N}.txt"
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
        file.write(f"alpha = {args.a}\n")
        file.write(f"gen = {args.gen}\n")
        file.write(f"N = {N}\n")
        file.write(f'M_SMBH = {args.Mbh}\n')
        file.write(f'Spin = {args.spin}\n')
        file.write(f'T = {args.T/(1e6)}\n')
        file.write(f"\n")
        file.write(f"Data:\n")
        file.write(f"m1/Msun, r0/Rg, cos_i, t_align/Myr, t_gw/Myr, t_migr/Myr, t_inspiral/Myr, is_emri, Ng, R_final/Rg, lisa_radii/Rg, t_lisa/Myr, t_final/Myr, lisa_flag, total_flags\n")
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
        N_iter = int(N)
        chunk_size = int(N_iter/N_batches)

        # run parallel simulations and print results in file
        #with multiprocessing.Pool(1) as pool:
        with multiprocessing.Pool(os.cpu_count()) as pool:
            with tqdm(total=N_iter) as pbar:
                for i in range(N_batches):
                    input_data = [(args, cluster_df, mass_prim_vk, r_pu_1g, disk, N) for _ in range(chunk_size)]
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
        print(f'file {file_name} closed.')
################################################################################################


