import os
from pathlib import Path
from make_yield_file_with_z import make_yield_file
import numpy as np

zform_min=0.05
zform_max=4.85
zform_step=0.05

SNR_lim=30

zforms = np.round(np.arange(zform_min, zform_max + 0.5 * zform_step, zform_step), 6, )

for BIMF in ["Vaccaro"]: #, "Bartos", "PY", "Tagawa", "Vaccaro"
    for RD in ["Rom"]: #, "Bartko", "Rom", "PY"
        for le in ["0.01"]:
            for Mbh in ['1.0e+04', '1.0e+05', '4.0e+05', '1.0e+06', '4.0e+06', '1.0e+07']: #'1.0e+04', '1.0e+05', '4.0e+05', '1.0e+06', '4.0e+06', '1.0e+07'
                for disk in ["SG"]: #, "NT"]:
                    for alpha in ["0.1"]: #, "0.01"]:
                        for spin in ["0.9"]: #, "0.99"]:
                            for Tdisk in ["10.0"]: #, "1.0", "100.0"]:
                                for wind in ["On"]: #, "Off", "Partial"]:
                                    for TT in ["G23"]: #, "P10"
                                        for zform in zforms:
                                            print(f"\n=== z_form = {zform:.3f} ===")
                                        
                                            infile = (
                                                    f"/Users/pmxks13/PhD/EMRIs_test/EMRI_Rates/"
                                                    f"{BIMF}/{RD}/le_{le}/Mbh_{Mbh}/{disk}/alpha_{alpha}/spin_{spin}/"
                                                    f"Tdisk_{Tdisk}/wind_{wind}/EMRIs_{TT}_1g_5.txt"
                                                )

                                            label=f'BIMF_{BIMF}/RD_{RD}/TT_{TT}/Tdisk_{Tdisk}/wind_{wind}'
                                            
                                            if not (os.path.exists(infile)):
                                                    print(f"SKIP missing event files: {infile}")
                                                    continue

                                            print(f'Processing infile: {infile}')

                                            make_yield_file(label, Path(infile), float(zform), SNR_lim)
                                            # try:
                                            #     make_yield_file(label, Path(infile), float(zform), SNR_lim)
                                            # except Exception as e:
                                            #     print(f"[FAILED] infile, zform {zform}")
                                    
