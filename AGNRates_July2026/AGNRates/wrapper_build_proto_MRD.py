#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from build_proto_MRD import run_proto_mrd, get_cosmo_cache


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--runs-base", required=True)
    p.add_argument("--alpha", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--pm-file", required=True)
    p.add_argument("--pfedd-file", required=True)
    p.add_argument("--redshift-model", required=True)

    p.add_argument("--zform-min", type=float, default=0.0)
    p.add_argument("--zform-max", type=float, default=15.0)
    p.add_argument("--zform-step", type=float, default=0.01)

    p.add_argument("--zmax", type=float, default=10.5)
    p.add_argument("--nz", type=int, default=105)
    p.add_argument("--outdir", default="../outputs/protoMRD")

    p.add_argument(
        "--yield-labels",
        nargs="+",
        default=["1g", "ng"],
    )

    return p.parse_args()


def main():
    args = parse_args()

    zforms = np.round(
        np.arange(args.zform_min, args.zform_max + 0.5 * args.zform_step, args.zform_step),
        6,
    )

    # Build cosmology once, large enough for all requested zform and z
    cosmo_zmax = max(args.zmax, float(np.max(zforms)), 15.0) + 2.0
    cosmo_cache = get_cosmo_cache(zmax=cosmo_zmax)

    print(
        f"Built cosmology cache once with zmax={cosmo_zmax:.2f}. "
        f"Now processing {len(zforms)} z_form values."
    )

    failed = []

    for zform in zforms:
        print(f"\n=== z_form = {zform:.3f} ===")

        try:
            run_proto_mrd(
                runs_base=Path(args.runs_base),
                alpha=Path(args.alpha),
                label=args.label,
                pm_file=args.pm_file,
                pfedd_file=args.pfedd_file,
                redshift_model=args.redshift_model,
                zform=float(zform),
                zmax=args.zmax,
                nz=args.nz,
                outdir=args.outdir,
                yield_labels=tuple(args.yield_labels),
                cosmo_cache=cosmo_cache,
            )
        except Exception as e:
            print(f"[FAILED] z_form={zform:.3f}: {e}")
            failed.append((float(zform), str(e)))

    print("\nDone.")
    if failed:
        print("Failures:")
        for zf, err in failed:
            print(f"  z_form={zf:.3f}: {err}")


if __name__ == "__main__":
    main()