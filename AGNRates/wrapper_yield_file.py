import os
from pathlib import Path
from make_yield_file import make_yield_file

for BIMF in ["PY"]: #, "Bartos", "Vaccaro", "Tagawa"
    for RD in ["PY"]: #, "Bartko", "Rom"
        for le in ["0.01"]:
            for Mbh in ['1.0e+04', '1.0e+05', '4.0e+05', '1.0e+06', '4.0e+06', '1.0e+07']:
                for disk in ["SG"]: #, "NT"]:
                    for alpha in ["0.1"]: #, "0.01"]:
                        for spin in ["0.9"]: #, "0.99"]:
                            for Tdisk in ["10.0"]: #, "1.0", "100.0"]:
                                for wind in ["On"]: #, "Off", "Partial"]:
                                    for TT in ["P10"]: #, "G23"
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
                                        try:
                                            make_yield_file(label, Path(infile))
                                        except Exception as e:
                                            print(f"[FAILED] infile")
                                
