import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sb

import binary_formation_distribution_V12 as myscript
import NT_disk_Eqns_V1 as jscript
import pagn.constants as ct

from datetime import datetime, timedelta
from tqdm import tqdm
from matplotlib.lines import Line2D
import pagn

def file_opener(filename):
    with open(filename) as f:
        lines = f.readlines()
    header_end = lines.index("Data:\n") + 1
    data = pd.read_csv(filename, delimiter=" ", skiprows=header_end)
    data.columns = [col.strip().replace(",", "") for col in data.columns]
    return data

def parse(value):
    try:
        if not '_' in value:
            return int(value)
        raise ValueError
    except ValueError:
        try:
            if not '_' in value:
                return float(value)
            raise ValueError
        except ValueError:
            try:
                if ':' in value:
                    parts = value.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds, microseconds = map(float, parts[2].split('.')) if '.' in parts[2] else (int(parts[2]), 0)
                        return timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=int(microseconds))
                return datetime.strptime(value, "%y%m%d_%H%M")
            except ValueError:
                return value

def load_file_2(filename):
    params = {}
    data = {}
    
    with open(filename, 'r') as file:
        for line in file:
            if line=="Parameters:\n": continue
            elif line=="\n": continue
            elif line=="Data:\n": break
            else:
                line_splitted = line.strip().split()
                if len(line_splitted)==3: 
                    params[line_splitted[0]] = parse(line_splitted[2])
                else: 
                    params[line_splitted[0]] = []
                    for i in range(2, len(line_splitted)): 
                        params[line_splitted[0]].append(parse(line_splitted[i]))

        headers = file.readline().strip().replace(",", "").split()
        # print(headers)
        data = {header: [] for header in headers}

        for line in file:
            values = line.strip().split()
            for i, header in enumerate(headers):
                data[header].append(parse(values[i]))
        
        for header in headers: data[header] = np.array(data[header])
    return params, data


filename = f"/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/EMRI_Rates_Summary_4.txt"
params = {}

with open(filename) as f:
    lines = f.readlines()
    for i in range(0, len(lines)):
        line=lines[i]
        if line==lines[0]:
            line_splitted = line.strip().split()
            for split in line_splitted:
                params[split] = []
        else:
            line_splitted = line.strip().split()
            for k in range(0, len(line_splitted)):
                try:
                    params[list(params)[k]].append(float(line_splitted[k]))
                except ValueError:
                    params[list(params)[k]].append(line_splitted[k])
    f.close()

params=pd.DataFrame.from_dict(params)

groups=params.groupby(['TT', 'BIMF', 'RD', 'wind', 'T_disc/Myr'], as_index=True) 

torques=True
mass_func=True
radial_dist=True

fig, ax = plt.subplots()

for group in groups:
    for i in range(0, len(params)):
        try:
            wind=group[1]['wind'][i]
            Tdisk=group[1]['T_disc/Myr'][i]
            TT=group[1]['TT'][i]
            Mbh=group[1]['MBH/Msun'][i]
            if Tdisk==10.0 and wind=='On':
                BIMF=group[1]['BIMF'][i]
                RD=group[1]['RD'][i]
                DT=group[1]['DT'][i]
                alpha=group[1]['alpha'][i]
                le=group[1]['le'][i]
                N_emri=group[1]['N_EMRI'][i]
                Tdisk=group[1]['T_disc/Myr'][i] * 1e6
                G_emri=N_emri*(1e9/Tdisk)
                # print(f'\nTT: {TT}, BIMF: {BIMF}, RD: {RD}, wind: {wind}')
                # print(f'Tdisk {Tdisk/1e6:.1f} Myr, {G_emri:.1e} NEmri/Gyr')
                if TT=='P10':
                    color='firebrick'
                elif TT=='B16':
                    continue
                elif TT=='G23':
                    color='gold'

                if BIMF=='PY':
                    marker='o'
                if BIMF=='Bartos':
                    continue
                if BIMF=='Tagawa':
                    marker='*'
                if BIMF=='Vaccaro':
                    marker='D'

                if RD=='PY':
                    fillstyle='full'
                if RD=='Bartko':
                    fillstyle='none'
                if RD=='Rom':
                    fillstyle='right'

                # if DT  == "SG":
                #     disk = pagn.SirkoAGN(Mbh=Mbh*jscript.MSun , alpha=alpha, le=le)
                #     disk.solve_disk()
                
                # Rmax=disk.Rmax

                # m=10
                # Rsch=2 * ct.G * Mbh*jscript.MSun / ct.c**2

                # mean_Gamma=myscript.compute_torque(disk, m*jscript.MSun , Mbh*jscript.MSun, TT, wind) 

                # # traps = myscript.mig_trap(disk, mean_Gamma) 
                # # innermost_trap = traps[0] if len(traps) > 0 else Rmax
                
                # antitraps = myscript.anti_trap(disk, mean_Gamma) 
                # innermost_antitrap = antitraps[0] if len(antitraps) > 0 else Rmax

                # print(f'\n**Mbh: {Mbh:1e} MSun**')
                # print(f'**Rmax: {Rmax/jscript.pc:1e} pc**')
                # print(f'**innermost antitrap: {innermost_antitrap/jscript.pc:1e} pc**\n')
                
                ax.plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.5, markeredgewidth=1.5)
        except KeyError:
            pass

legend_rd = [Line2D([0], [0], marker='o', color='k', label='WeakPowerLaw',linestyle='None', fillstyle='full'), 
                Line2D([0], [0], marker='o', color='k', label='StrongPowerLaw',linestyle='None', fillstyle='none'),
                Line2D([0], [0], marker='o', color='k', label='Segmented',linestyle='None', fillstyle='right')]

legend_TT =[Line2D([0], [0], marker='o', color='firebrick', label='Isothermal (P10)',linestyle='None'),
        #    Line2D([0], [0], marker='o', color='darkorange', label='B16',linestyle='None'),
            Line2D([0], [0], marker='o', color='gold', label='+3DThermal (G24)',linestyle='None')]

legend_bimf = [Line2D([0], [0], marker='o', color='k', label='TopHeavy',linestyle='None'),
                #    Line2D([0], [0], marker='p', color='k', label='Bartos',linestyle='None'),
            Line2D([0], [0], marker='*', color='k', label='ProgeniterGen',linestyle='None'),
            Line2D([0], [0], marker='D', color='k', label='PopSynth',linestyle='None')]


Mbh_PY=[1e5, 4e5, 1e6, 4e6, 1e7]
Rates_PY=[4.2e3, 1.7e5, 4.5e5, 2.4e5, 1.3e5]

ax.plot(Mbh_PY, Rates_PY, marker='.', color='black', markersize=10, linestyle='None')

ax.set_xscale('log')
ax.set_yscale('log')

# ax.set_xlim(9e4, 2e7)
ax.set_ylim(30, 2e7)

ax.set_ylabel('$N_{EMRI}/Gyr$')
ax.set_xlabel('SMBH $[M_{\odot}]$')

tt_legend=ax.legend(title='  TT', handles=legend_TT, loc='upper left', bbox_to_anchor=(1.02, 1.0), alignment='left')

bimf_legend=ax.legend(title='  BIMF', handles=legend_bimf, bbox_to_anchor=(1.02, 0.8), loc='upper left', alignment='left')

rd_legend=ax.legend(title='  RD', handles=legend_rd, loc='upper left', bbox_to_anchor=(1.02, 0.54), alignment='left')

legends=tt_legend, bimf_legend, rd_legend
for legend in legends:
    legend._legend_box.width = 550
    ax.add_artist(legend)

plt.savefig(f'/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/rates_updated.png', dpi=300, bbox_inches = "tight")