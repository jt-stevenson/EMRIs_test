import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

legend_rd = [Line2D([0], [0], marker='o', color='k', label='WeakPowerLaw',linestyle='None', fillstyle='full', markersize=10), 
                Line2D([0], [0], marker='o', color='k', label='StrongPowerLaw',linestyle='None', fillstyle='none', markersize=10),
                Line2D([0], [0], marker='o', color='k', label='Segmented',linestyle='None', fillstyle='right', markersize=10)]

legend_TT =[Line2D([0], [0], marker='o', color='firebrick', label='Isothermal (P10)',linestyle='None', markersize=10),
            Line2D([0], [0], marker='o', color='olivedrab', label='+3DThermal (G24)',linestyle='None', markersize=10)]

legend_bimf = [Line2D([0], [0], marker='o', color='k', label='TopHeavy',linestyle='None', markersize=10),
            Line2D([0], [0], marker='p', color='k', label='ProgeniterGen',linestyle='None', markersize=10),
            Line2D([0], [0], marker='D', color='k', label='PopSynth',linestyle='None', markersize=10)]

fig, ax = plt.subplots(3, 1, figsize=(6, 12), dpi=300, sharex=True)

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0.25)
fig.subplots_adjust(top=0.93)

for i in range(0, len(params)):
    try:
        Mbh=params['MBH/Msun'][i]
        wind=params['wind'][i]
        Tdisk=params['T_disc/Myr'][i]
        TT=params['TT'][i]
        BIMF=params['BIMF'][i]
        RD=params['RD'][i]
        N_emri=params['N_EMRI'][i]

        if Tdisk==10.0 and wind=='On':
            Tdisk=params['T_disc/Myr'][i] * 1e6
            G_emri=N_emri*(1e9/Tdisk)

            if TT=='P10':
                color='firebrick'
            elif TT=='G23':
                color='olivedrab'

            if BIMF=='PY':
                marker='o'
            if BIMF=='Tagawa':
                marker='p'
            if BIMF=='Vaccaro':
                marker='D'

            if RD=='PY':
                fillstyle='full'
            if RD=='Bartko':
                fillstyle='none'
            if RD=='Rom':
                fillstyle='right'

            if BIMF=='Vaccaro':
                if RD=='Rom':
                    if TT=='P10':
                        ax[0].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                        
                    elif TT=='G23':
                        ax[0].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                        
                    
            if TT=='G23':
                if RD=='Rom':
                    if BIMF=='PY':
                            ax[1].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                            
                    if BIMF=='Tagawa':
                            ax[1].plot(Mbh, G_emri, marker=marker, color=color, markersize=10,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                            
                    if BIMF=='Vaccaro':
                            ax[1].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                            
                    
            if TT=='G23':
                if BIMF=='Vaccaro':
                    if RD=='PY':
                        ax[2].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                                    
                    if RD=='Bartko':
                        ax[2].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                                    
                    if RD=='Rom':
                        ax[2].plot(Mbh, G_emri, marker=marker, color=color, markersize=8,  fillstyle=fillstyle, alpha=0.8, markeredgewidth=1.5)
                                    
                    
    except KeyError:
        pass

Mbh_PY=[1e5, 4e5, 1e6, 4e6, 1e7]
Rates_PY=[4.2e3, 1.7e5, 4.5e5, 2.4e5, 1.3e5]

for axes in ax:
    axes.plot(Mbh_PY, Rates_PY, marker='.', color='black', markersize=10, linestyle='None')
    axes.set_xscale('log')
    axes.set_yscale('log')
    # ax.set_xlim(9e4, 2e7)


ax[1].set_ylabel('$N_{EMRI}/Gyr$')

ax[0].set_ylim(1e3, 2e6)
ax[1].set_ylim(1e3, 2e6)
ax[2].set_ylim(1e2, 2e6)

ax[2].set_xlabel('SMBH $[M_{\odot}]$')

ax[0].legend(title='  TT', handles=legend_TT, loc='upper left', alignment='left')

ax[1].legend(title='  BIMF', handles=legend_bimf, loc='upper left', alignment='left')

ax[2].legend(title='  RD', handles=legend_rd, loc='upper left', alignment='left')

plt.savefig(f'/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/rates_updated_sep.png', dpi=300, bbox_inches = "tight")

fig, ax = plt.subplots()

for i in range(0, len(params)):
    try:
        wind=params['wind'][i]
        Tdisk=params['T_disc/Myr'][i]
        TT=params['TT'][i]
        Mbh=params['MBH/Msun'][i]
        if Tdisk==10.0 and wind=='On':
            BIMF=params['BIMF'][i]
            RD=params['RD'][i]

            N_emri=params['N_EMRI'][i]
            Tdisk=params['T_disc/Myr'][i] * 1e6
            G_emri=N_emri*(1e9/Tdisk)
            markersize=8

            if TT=='P10':
                color='firebrick'
            elif TT=='G23':
                color='olivedrab'

            if BIMF=='PY':
                marker='o'
            if BIMF=='Tagawa':
                marker='p'
                markersize=10
            if BIMF=='Vaccaro':
                marker='D'

            if RD=='PY':
                fillstyle='full'
            if RD=='Bartko':
                fillstyle='none'
            if RD=='Rom':
                fillstyle='right'

            ax.plot(Mbh, G_emri, marker=marker, color=color, markersize=markersize,  fillstyle=fillstyle, alpha=0.5, markeredgewidth=1.5)
    except KeyError:
        pass

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