
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

printing=True
plotting=True
type_II_computation = "conservative" 
C=663176
Spin=0.9

Fixed=True

import binary_formation_distribution_V8 as myscript
import NT_disk_Eqns_V1 as jscript

################################################################################################
### Simulation routine #########################################################################
################################################################################################xs
def iteration(args, MBH, T, mass_sec, mass_prim_vk, r_pu_1g):
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
    Rg = ct.G*Mbh/ct.c**2
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

    # inspiral happens within disk's lifetime
    emri_within_T = min(t_migr, t_gw) < T

    # for EMRIs this is not a binary quantity but depends only on SMBH spin (assumed random)
    # spin=np.random.rand()
    # future task - add SMBH spin distribution to draw from?
    chi_eff = 2 * spin - 1  # in [-1, 1]

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

    M1=np.array([m1])

    gap = np.full(M1.shape, 'type_I', dtype='<U7')

    Gammas = myscript.compute_torque_function(args, disk, m1, Mbh)
    Gammas_GW = jscript.compute_torque_function(args, disk, m1, Mbh)

    gap_event_1 = myscript.type_II_event(disk, m1, Mbh)
    R_event_1=jscript.r_isco_event(Mbh, m1, spin)

    print(f'SMBH {Mbh/ct.MSun:.3e} Msun and SBH {m1/ct.MSun:.3e} Msun, r0 {r0/Rg:.3e} Rg')

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

    #check whether "events" happened (reaching Type I migration trap or doing Type II migration, or crossing inner edge of disk)

    # if you open a gap
    # keep integrating with Type II torque
    if len(solution1.t_events[1]) >0:
        print("SBH does type II")
        gap[0]='type_II'
        t_gap = solution1.t_events[1][0]
        r_gap = solution1.y_events[1][0][0]
        # Stage 2 : Type II migration
        if (type_II_computation == "conservative") or (type_II_computation == "conservative+extra_h_condition"):
            solution1_TypeII = solve_ivp(myscript.rdot_typeII, [t_gap, T], [r_gap],
                                     args=(disk,),
                                     first_step=1e3*ct.yr, rtol=1e-3, atol=0.0)
        elif type_II_computation == "modified_type_I":
            solution1_TypeII = solve_ivp(myscript.rdot_typeII_Kanagawa2018, [t_gap, T], [r_gap],
                                     args=(M[0], disk, Mbh,),
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
                    t4 = np.append(t3, T)
                    r4 = np.append(r3, r3[-1])

                    t3=np.concatenate((t3, t4))
                    r3=np.concatenate((r3, r4))

                t2 = np.concatenate((t2, t3))
                r2 = np.concatenate((r2, r3))

            elif len(solution1_lisa_entry.t_events[1])>0 and len(solution1_lisa_entry.t_events[0])==0: #check if sbh crosses SMBH event horizon if LISA band not left
                print('SBH has crossed EH')
                t2 = np.append(t2, T)
                r2 = np.append(r2, r2[-1])

                # t2 = np.concatenate((t2, t3))
                # r2 = np.concatenate((r2, r3))

            t1 = np.concatenate((t1, t2))
            r1 = np.concatenate((r1, r2))
        
        elif len(solution1.t_events[2])==0 and len(solution1.t_events[3])>0: #check if sbh crosses SMBH event horizon if lisa band never entered
            print('SBH has crossed EH')
            t1 = np.append(t1, T)
            r1 = np.append(r1, r1[-1])

    #Jupiter Original Code...

    #We want to find the location of the secondary (SBH) when the disc disperses to check if the signal produced is in the LISA band
    t_final=t1[len(t1)-1]
    r_final=r1[len(r1)-1]

    final_time=20*T

    print(f'input for extension {t_final}, {final_time}, {r_final}')

    if emri_within_T==False and emri_flag==True:
        print('Running trajectory beyond disc evaporation to check for "dried" EMRI...')
        solution1_extended = solve_ivp(fun=myscript.rdot, 
                            t_span=[t_final, final_time], 
                            y0=[r_final],
                            method='RK23', 
                            events=(jscript.LISA_band_enter, R_event_1),
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas_GW, Mbh, traps))
        t1 = solution1.t
        r1 = solution1.y[0]
        
        if solution1_extended.t_events[0]>0: #enters LISA band
            lisa_entry_radii=solution1_extended.y_events[0][0][0]
            t_lisa_entry=solution1_extended.t_events[0][0]

            lisa_flag=4

            print(f"SBH has entered LISA band at {t_lisa_entry/(1e6*ct.yr):.3e} Myr, after Tdisc {T/(1e6*ct.yr):.3e} Myr")
            
            solution1_extended_entry = solve_ivp(fun=myscript.rdot, 
                            t_span=[t_lisa_entry, final_time], 
                            y0=[lisa_entry_radii],
                            method='RK23', 
                            events=(jscript.LISA_band_exit, R_event_1),
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas_GW, Mbh, traps))
            
            t2=solution1_extended_entry.t[1:]
            r2=solution1_extended_entry.y[0][1:]

            if solution1_extended_entry.t_events[0]>0: #exit LISA band
                lisa_exit_radii=solution1_extended_entry.y_events[0][0][0]
                t_lisa_exit=solution1_extended_entry.t_events[0][0]
                print(f"SBH has exited LISA band at {t_lisa_exit/(1e6*ct.yr):.3e} Myr, after Tdisc {T/(1e6*ct.yr):.3e} Myr")

                solution1_extended_exit = solve_ivp(fun=myscript.rdot, 
                            t_span=[t_lisa_entry, final_time], 
                            y0=[lisa_entry_radii],
                            method='RK23', 
                            events=(R_event_1),
                            first_step=1e3*ct.yr,
                            rtol=1e-3,
                            atol=1e-9,
                            args=(m1, Gammas_GW, Mbh, traps))
                
                t3=solution1_extended_exit.t[1:]
                r3=solution1_extended_exit.y[0][1:]

                if solution1_extended_exit.t_events[0]>0: #reaches isco after exiting LISA band
                    print(f"SBH merged at {t3[len(t3)-1]/(1e6*ct.yr):.3e} Myr, after Tdisc {T/(1e6*ct.yr):.3e} Myr")
                    t4 = np.append(t3, T)
                    r4 = np.append(r3, r3[-1])

                    t3=np.concatenate((t3, t4))
                    r3=np.concatenate((r3, r4))

                t2 = np.concatenate((t2, t3))
                r2 = np.concatenate((r2, r3))

            elif len(solution1_extended_entry.t_events[1])>0 and len(solution1_extended_entry.t_events[0])==0: #check if sbh crosses SMBH event horizon if LISA band not left
                print(f"SBH merged at {t2[len(t2)-1] /(1e6*ct.yr):.3e} Myr, after Tdisc {T/(1e6*ct.yr):.3e} Myr")
                t2 = np.append(t2, final_time)
                r2 = np.append(r2, r2[-1])

            t1 = np.concatenate((t1, t2))
            r1 = np.concatenate((r1, r2))

        elif len(solution1_extended.t_events[0])==0 and len(solution1_extended.t_events[1])>0: #check if sbh crosses SMBH event horizon if lisa band never entered
            print(f"SBH merged at {t1[len(t1)-1]/(1e6*ct.yr):.3e} Myr, after Tdisc {T/(1e6*ct.yr):.3e} Myr")           
            t1 = np.append(t1, final_time)
            r1 = np.append(r1, r1[-1])

        t_final=t1[len(t1)-1]
        r_final=r1[len(r1)-1]

    rG=ct.G*Mbh*(1/(ct.c*ct.c))

    if t_final!=T and printing==True:
        print(f'SOMETHING HAS GONE WRONG! T={T/(1e6*ct.yr):.3e}!=t_final={t_final/(1e6*ct.yr):.3e}')
    
    if printing==True:
        print(f'For SMBH {Mbh/ct.MSun:.3e} Msun and SBH {m1/ct.MSun:.3e} Msun with r0 {r0/rG:.3e} Rg, At time T={T/(1e6*ct.yr):.3e}={t_final/(1e6*ct.yr):.3e} Myrs, R_final={r_final/rG:.3e} Rg')
        print(f"emri flag {emri_flag}, emri within T {emri_within_T}")
    M=Mbh+m1

    # lisa_flag, lisa_radii=jscript.LISAband_flag(r0, r_final, Mbh, m1)
    lisa_radii=lisa_entry_radii

    if t_lisa_exit==0:
        t_lisa=t_lisa_entry-t_final
    elif t_lisa_exit!=0:
        t_lisa=t_lisa_entry-t_lisa_exit


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
    if Fixed==True:
        return f"{m1/ct.MSun:.3e} {r0/rG:.3e} {t_gw/(1e6*ct.yr):.3e} {t_migr/(1e6*ct.yr):.3e} {is_emri} {Ng} {r_final/rG:.3e} {lisa_radii/rG:.3e} {t_lisa/(1e6*ct.yr):.3e} {t_final/(1e6*ct.yr):.3e} {lisa_flag} {flags}\n"
    elif Fixed==False:
        return f"{np.log10(Mbh/ct.MSun):.1f} {m1/ct.MSun:.3e} {r0/rG:.3e} {chi_eff:.3e} {T/(1e6*ct.yr):.3e} {t_gw/(1e6*ct.yr):.3e} {t_migr/(1e6*ct.yr):.3e} {is_emri} {Ng} {r_final/rG:.3e} {lisa_radii/rG:.3e} {lisa_exit_radii/rG:.3e} {t_lisa/(1e6*ct.yr):.3e} {t_final/(1e6*ct.yr):.3e} {lisa_flag} {flags}\n"
    
################################################################################################
### Read parameters from input #################################################################
################################################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-DT', type=str, default="SG", choices=['SG', 'TQM'])
    parser.add_argument('-TT', type=str, default="G23", choices=['B16', 'G23'])
    parser.add_argument('-gen', type=str, default='1g', choices=['1g', 'Ng'])
    parser.add_argument('-a', type=float, default=0.01)    # real number
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


