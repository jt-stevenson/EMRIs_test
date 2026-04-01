
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
    T = args.T * ct.yr # disk lifetime
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

    # compute trap locations 
    antitraps = myscript.anti_trap(disk, Gamma(disk.R)) 
    innermost_antitrap = antitraps[0] if len(antitraps) > 0 else Rmax
    
    # time and initial radius
    t0 = 0
    r0=float(cluster_df['r [Rg]'][a]*R_g)
    cos_i=float(cluster_df['cos_i'][a])

    if args.gen=='Ng':
        b = np.random.randint(0, len(r_pu_1g))
        r1 = r_pu_1g[b]
        r0 = myscript.pos_after_kick(r1, mass_prim_vk, Mbh)
    
    # select as EMRI if inside innermost trap, migrating inward
    Gamma_r0 = Gamma(r0)
    emri_flag = (Gamma_r0 < 0) and (r0 < innermost_antitrap) #NB: this condition is no longer enforced in extraction of r0

    # Alignment timescale, Rowan et al 2024
    f=interp1d(disk.R, disk.h, kind='linear', fill_value='extrapolate')
    h=f(r0)
    t_align = jscript.T_align(disk, Mbh, m1, cos_i, h, r0)

    # Encounter timescale, Rowan et al 2024
    # print(f'Mbh: {Mbh}, m1: {m1}, r0: {r0}, cos_i: {cos_i}, N: {N}, disk: {disk}')
    t_enc = jscript.T_enc(Mbh, m1, r0, N, disk)

    # Migration timescale = L / |Gamma|
    L = m1 * np.sqrt(ct.G * Mbh * r0)
    t_migr = L / np.abs(Gamma_r0)

    # GW inspiral time
    t_gw = (5 / 256) * (ct.c**5 / (ct.G**3)) * (r0**4) / (m1 * Mbh**2)

    # alignment and inspiral happen within disk's lifetime + sbh not scattered faster than alignment

    # print(f't_align: {float(t_align)/(365*24*60*60*1e6)} Myr, t_enc: {float(t_enc)/(365*24*60*60*1e6)} Myr')
    align = t_align < T
    not_scattered = t_align < t_enc
    emri_within_T = min(t_migr, t_gw) < T

    # final flag
    is_emri = emri_flag and emri_within_T and align and not_scattered

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
        print(f'* EMRI FOUND! * Starting SBH {m1/ct.MSun:.3e} Msun, r0 {r0/R_g:.3e} Rg...\n')
    elif is_emri==False:
    #     is_EMRI=0
    #     # print(f'SBH {m1/ct.MSun:.3e} Msun, r0 {r0/R_g:.1e} Rg is not an EMRI, trap at {innermost_trap/R_g:.1e} Rg, antitrap at {innermost_antitrap/R_g:.1e} Rg, {emri_flag} and {emri_within_T} and {align} and {not_scattered}, quitting...', end='\r')
        print(f'SBH {m1/ct.MSun:.3e} Msun, r0 {r0/R_g:.1e} Rg is not an EMRI, antitrap at {innermost_antitrap/R_g:.1e} Rg, quitting...', end='\r')
        return

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


    GWf=jscript.GW_freq_fn(r0, Mbh, m1)
    if 1.0>GWf>0.0001:
        lisa_radii=r0
        lisa_flag=4

    # print(f'beginning solve SBH {m1/ct.MSun:.3e} Msun...')

    solution1 = solve_ivp(fun=myscript.rdot, 
                            t_span=[t0, T], 
                            y0=[r0],
                            method='RK23', 
                            events=(myscript.trap_dist, gap_event_1, jscript.LISA_band_enter, R_event_1), #, Rs_event_1)
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas, Mbh, traps))

    t1 = solution1.t
    r1 = solution1.y[0]

    # print(f'first integration complete for SBH {m1/ct.MSun:.3e} Msun')

    #check whether "events" happened (reaching Type I migration trap or doing Type II migration, or crossing inner edge of disk or if LISA band entered/exited)

    # if you open a gap
    # keep integrating with Type II torque
    if len(solution1.t_events[1])>0:
        # print("SBH does type II")
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
        # print("integration type II concluded")
        # Stitch the two segments 
        t1 = np.concatenate((t1, solution1_TypeII.t[1:]))     # skip duplicate t_gap
        r1 = np.concatenate((r1, solution1_TypeII.y[0][1:]))
    
    elif len(solution1.t_events[0])>0:  #checks if trap is reached- shouldn't happen due to conditions on r0, but want to keep incase we begin integrating from further out than inside the first trap
        t1 = np.append(t1, T)           #or for some other weirdness
        r1 = np.append(r1, r1[-1])
    
    else: 
        if len(solution1.t_events[2])>0: #check if sbh has entered the LISA band
            # print('SBH has entered LISA band')
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
                # print('SBH has left LISA band')
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
                    # print('SBH has crossed EH')
                    t_inspiral=t3[len(t3)-1]

                    t4 = np.append(t3, T)
                    r4 = np.append(r3, r3[-1])

                    t3=np.concatenate((t3, t4))
                    r3=np.concatenate((r3, r4))

                t2 = np.concatenate((t2, t3))
                r2 = np.concatenate((r2, r3))

            elif len(solution1_lisa_entry.t_events[1])>0 and len(solution1_lisa_entry.t_events[0])==0: #check if sbh crosses SMBH event horizon if LISA band not left
                # print('SBH has crossed EH')
                t_inspiral=t2[len(t2)-1]

                t2 = np.append(t2, T)
                r2 = np.append(r2, r2[-1])

                # t2 = np.concatenate((t2, t3))
                # r2 = np.concatenate((r2, r3))

            t1 = np.concatenate((t1, t2))
            r1 = np.concatenate((r1, r2))
        
        elif len(solution1.t_events[2])==0 and len(solution1.t_events[3])>0: #check if sbh crosses SMBH event horizon if lisa band never entered
            # print('SBH has crossed EH')
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

    if lisa_flag==0:
        lisa_bool=False
    if lisa_flag==4:
        lisa_bool=True
    
    # if printing==True:
    #     print(f'For SMBH {Mbh/ct.MSun:.3e} Msun and SBH {m1/ct.MSun:.3e} Msun with r0 {r0/rG:.3e} Rg (Rmin: {Rmin/R_g:.3e} Rg, Rmax: {Rmax/R_g:.3e} Rg) \nAt time T={T/(1e6*ct.yr):.3e}={t_final/(1e6*ct.yr):.3e} Myrs, R_final={r_final/rG:.3e} Rg\nemri flag {emri_flag}, emri within T {emri_within_T}, lisa detected? {lisa_bool}')
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
    # final_flag=is_EMRI+lisa_flag

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
    # print('FINISHED, WRITING TO FILE')
    return f"{m1/ct.MSun:.3e} {r0/rG:.3e} {cos_i:.3f} {t_align/(1e6*ct.yr):.3e} {t_gw/(1e6*ct.yr):.3e} {t_migr/(1e6*ct.yr):.3e} {t_inspiral/(1e6*ct.yr):.3e} {Ng} {r_final/rG:.3e} {lisa_radii/rG:.3e} {t_lisa/(1e6*ct.yr):.3e} {t_final/(1e6*ct.yr):.3e} {lisa_flag} {flags}\n"
################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM', 'NT'])
    parser.add_argument('-TT', type=str, default="B16", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-BIMF', type=str, default="PY", choices=['Vaccaro', 'Tagawa', 'Bartos', 'PY'])
    parser.add_argument('-RD', type=str, default="PY", choices=['Bartko', 'Rom', "PY"])
    parser.add_argument('-wind', type=str, default="On", choices=['On', 'Off', "Partial"])
    parser.add_argument('-a', type=float, default=0.1)
    parser.add_argument('-le', type=float, default=0.01)
    parser.add_argument('-spin', type=float, default=0.9)  # real number
    parser.add_argument('-Mbh', type=float, default=1e7)   # MSun
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

    MBH_power=int(np.log10(args.Mbh))

    if args.DT  == "SG":
        disk = pagn.SirkoAGN(Mbh=Mbh, alpha=alpha, le=le)
        Rmax = disk.Rmax
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
    print('solving disk...')
    disk.solve_disk()

    print('initialising cluster...')
    T_clust= 10**6 * ct.yr
    # try:
    #     cluster_df=pd.read_csv(f'EMRI_Rates/{args.BIMF}/dataframes/{args.DT}_1e{MBH_power}_alpha_{args.a}_le_{args.le}_spin_{args.spin}_N_*.csv')
    # except FileNotFoundError:
    #     print('cluster_df not found, sampling cluster...')
    #     cluster_df=jscript.cluster_sampling(Mbh, args.a, args.spin, args.le, args.DT, args.BIMF, disk, save=True)
    if args.RD=='Bartko':
        gamma=-2.5
    elif args.RD=='PY':
        gamma=-1.8
    cluster_df=jscript.cluster_sampling(Mbh, args.a, args.spin, args.le, args.DT, args.BIMF, args.RD, disk, T_clust, gamma=gamma, save=True)
    
    N=len(cluster_df)
    print(f"N: {N}")

    jscript.plot_torques(args, disk, Mbh, cluster_df['mbh [Msun]'], T_clust)

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

        dir_name = f"/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/"
        subdir_name=dir_name+f'{args.BIMF}/Mbh_{args.Mbh:.1e}/{args.DT}/alpha_{args.a}/spin_{args.spin}/Tdisk_{args.T/(1e6)}/wind_{args.wind}/'
        if not os.path.exists(subdir_name):
            os.makedirs(subdir_name)
        file_name = subdir_name+f"EMRIs_{args.TT}_{args.gen}_{N}.txt"
        print(f'file name: {file_name}')
        file_name_1g = dir_name+f"EMRIs_{args.TT}_1g_{N}_3.txt"
        if args.gen=='Ng' and  not os.path.exists(file_name_1g):
            print()
            print('There is no 1g source for this Ng simulation. Run the same simulation for 1g first!')
            quit()
        file = open(file_name, "w")

        print('writing params to file...')

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
        file.write(f"m1/Msun, r0/Rg, cos_i, t_align/Myr, t_gw/Myr, t_migr/Myr, t_inspiral/Myr, Ng, R_final/Rg, lisa_radii/Rg, t_lisa/Myr, t_final/Myr, lisa_flag, total_flags\n")

        print('params written to file...')
        file.close()
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
                    print(f'N_batch: {i}')
                    file = open(file_name, "a")
                    input_data = [(args, cluster_df, mass_prim_vk, r_pu_1g, disk, N) for _ in range(chunk_size)]
                    results = pool.starmap(iteration, input_data)
                    print(f'\n N_batch {i} complete, writing to file')
                    if len(results) != chunk_size:
                        print(f"[Warning] Batch {i} returned {len(results)} runs instead of {chunk_size}")
                    for result in results:
                        if result==None:
                            pass
                        else:
                            file = open(file_name, "a")
                            file.writelines(result)
                            file.close()
                    pbar.update(chunk_size)
                    print('Next Chunk...')
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

################################################################################################
### Calculate EMRI Rates for the SMBH  #########################################################
################################################################################################
        data=myscript.load_file(file_name)
        N_emri=len(data[1]['m1/Msun,'])
        
        print('printing following to summary file...')
        print(f'MBH: {args.Mbh:.1e} MSun\nSpin: {args.spin}\nalpha: {args.a}\nle: {args.le}\nwind: {args.wind}\nTdisk: {args.T/1e6:.1f} Myrs\nDT: {args.DT}\nTT: {args.TT}\nBIMF: {args.BIMF}\nRD: {args.RD}\nN: {N}, N_emri: {N_emri}\n')

        summary_file = dir_name+f"EMRI_Rates_Summary_2.txt"
        file = open(summary_file, 'a')
        file.write(f'{args.Mbh:.1e} {args.spin} {args.a} {args.le} {args.wind} {args.T/1e6:.1f} {args.DT} {args.TT} {args.BIMF} {args.RD} {N} {N_emri}\n')
        file.close()
        print(f'file {summary_file} closed, beginning next permutation.')
################################################################################################
