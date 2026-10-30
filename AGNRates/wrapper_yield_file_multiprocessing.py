import os
from pathlib import Path
from make_yield_file_with_z import make_yield_file
import numpy as np
import itertools
import multiprocessing

zform_min=0.05
zform_max=2.85
zform_step=0.05

SNR_lims=[20]

zforms = np.round(np.arange(zform_min, zform_max + 0.5 * zform_step, zform_step), 6, )
MBHs=['4.0e+06', '1.0e+07'] #'1.0e+04', '1.0e+05', '4.0e+05', '1.0e+06', '4.0e+06', '1.0e+07'
BIMFs=["Vaccaro"] #, "Bartos", "PY", "Tagawa", "Vaccaro"
RDs= ["Rom"] #, "Bartko", "Rom", "PY"
les=["0.01"]
disks=["SG"] #, "NT"]:
alphas=["0.1"] #, "0.01"]:
spins=["0.9"] #, "0.99"]:
Tdisks=["10.0"] #, "1.0", "100.0"]:
winds=["On"] #, "Off", "Partial"]:
TTs=["G23"] #, "P10"

def yield_wrapper(Mbh, BIMF , RD, le, disk, alpha, spin, Tdisk, wind, TT, zform, SNR_lim):
    infile = (f"/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/"
                f"{BIMF}/{RD}/le_{le}/Mbh_{Mbh}/{disk}/alpha_{alpha}/spin_{spin}/"
                f"Tdisk_{Tdisk}/wind_{wind}/EMRIs_{TT}_1g_5.txt")
    label=f'BIMF_{BIMF}/RD_{RD}/TT_{TT}/Tdisk_{Tdisk}/wind_{wind}'
    print(f'Processing infile: {infile}\n')
    make_yield_file(label, Path(infile), float(zform), SNR_lim)
    print(f'infile {infile} complete, outfile at: /Users/pmxks13/PhD/EMRIs_test/AGNRates/yields_files/Z0.02/SG_alpha_{alpha}/SNRs_{SNR_lim}/{label}/M_{Mbh}_fEdd_0.01/z_{zform}/yield_1g.txt\n')

if __name__ == '__main__':
    inlist = np.array(list(itertools.product(MBHs, BIMFs , RDs, les, disks, alphas, spins, Tdisks, winds, TTs, zforms, SNR_lims)))
    print(type(inlist), inlist.__sizeof__())
    N_batches = len(zforms)
    N_iter = int(len(inlist))

    chunk_size = int(N_iter/N_batches)
    print('multiprocessing starting...')
    for i in range(0, N_batches):
        print(f'N_batch: {i}')
        inlist_chunk=inlist[(i*chunk_size): ((i+1)*chunk_size)]
        with multiprocessing.Pool(2) as pool:
            results = pool.starmap(yield_wrapper, inlist_chunk)
        print(f'\n N_batch {i} complete, next chunk...')
