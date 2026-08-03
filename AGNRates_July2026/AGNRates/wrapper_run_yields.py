import os
import subprocess
import numpy as np


for disk in ["SG"]:
    for alpha in ["0.1", "0.01"]:
        for Mbh in [f"{x:.1f}" for x in np.arange(5.0, 9.0, 0.1)]:
            for fEdd in ["0.001", "0.01", "0.1", "1.", "10."]:
                for torque in ["B16_K18"]:
                    for sigma in ["0.0"]: #, "0.1"]:
                        for progr in ["agnostic"]: #, "all_prograde", "all_retrograde"]:
                            for pairup in ["differential_migration"]:
                                for gh in ["IG25"]: #, "IG20", "Calcino23"]:
                                    for lifetime in ["1."]:

                                        run_dir = (
                                            f"/gpfs/bwfor/work/ws/hd_tn184-AGN_fastcluster_grid/"
                                            f"RUNS_chi_zero/{disk}/alpha_{alpha}/logM_{Mbh}/fEdd_{fEdd}/"
                                            f"{torque}-3bb_{sigma}-{gh}-{progr}-tau_x_{lifetime}/outputs/"
                                        )

                                        first_file = os.path.join(run_dir, "first_gen.txt")
                                        nth_file   = os.path.join(run_dir, "nth_generation.txt")

                                        if not (os.path.exists(first_file) or os.path.exists(nth_file)):
                                            print(f"SKIP missing both event files: {run_dir}")
                                            continue

                                        cmd = [
                                            "python",
                                            "AGNRates/run_yields.py",
                                            "--run-dir", run_dir,
                                        ]

                                        print("RUN", " ".join(cmd))
                                        result = subprocess.run(cmd)

                                        if result.returncode != 0:
                                                print(f"FAILED: {run_dir}")