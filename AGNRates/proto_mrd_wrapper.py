#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from proto_mrd import (
    VALID_METALLICITY_MODELS,
    get_cosmo_cache,
    run_proto_mrd,
)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--yields-base",
        dest="yields_base",
        required=True,
        help=(
            "Metallicity-dependent yield-root template, e.g. "
            "'/path/outputs/yields/Z{Z}'"
        ),
    )
    p.add_argument("--alpha", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--pm-file", required=True)
    p.add_argument("--pfedd-file", required=True)
    p.add_argument("--redshift-model", required=True)
    p.add_argument("--metallicity-model", required=True, choices=sorted(VALID_METALLICITY_MODELS))
    p.add_argument(
        "--metallicity-values",
        nargs="+",
        default=None,
        help="Optional explicit Z grid; otherwise discover values from --yields-base",
    )
    p.add_argument(
        "--metallicity-value",
        type=float,
        default=None,
        help="Single explicit Z value required when --metallicity-model single_Z",
    )

    p.add_argument("--zform-min", type=float, default=0.0)
    p.add_argument("--zform-max", type=float, default=4.85)
    p.add_argument("--zform-step", type=float, default=0.05)

    p.add_argument("--zmax", type=float, default=4.5)
    p.add_argument("--nz", type=int, default=45)
    p.add_argument("--SNR-lim", type=int, default=30)
    p.add_argument("--outdir", default="outputs/protoMRD")

    p.add_argument(
        "--yield-labels",
        nargs="+",
        default=["1g", "ng"],
        help="Optional yield labels; by default, read labels from summary_yields.txt",
    )

    return p.parse_args()


def main():
    args = parse_args()

    metallicity_values = args.metallicity_values
    if args.metallicity_model == "single_Z":
        if args.metallicity_value is not None and metallicity_values is not None:
            raise ValueError("Use either --metallicity-value or --metallicity-values, not both")
        metallicity_values = (
            [args.metallicity_value]
            if args.metallicity_value is not None
            else metallicity_values
        )
        if metallicity_values is None or len(metallicity_values) != 1:
            raise ValueError(
                "single_Z requires exactly one value via --metallicity-value"
            )
    elif args.metallicity_value is not None:
        raise ValueError("--metallicity-value is only valid with --metallicity-model single_Z")

    zforms = np.round(
        np.arange(args.zform_min, args.zform_max + 0.5 * args.zform_step, args.zform_step),
        6,
    )

    # Build cosmology once, large enough for all requested zform and z
    cosmo_zmax = max(args.zmax, float(np.max(zforms)), 4.5) + 2.0
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
                yields_base=Path(args.yields_base),
                alpha=Path(args.alpha),
                label=args.label,
                pm_file=args.pm_file,
                pfedd_file=args.pfedd_file,
                redshift_model=args.redshift_model,
                metallicity_model=args.metallicity_model,
                zform=float(zform),
                zmax=args.zmax,
                nz=args.nz,
                SNR_lim=args.SNR_lim,
                outdir=args.outdir,
                yield_labels=tuple(args.yield_labels),
                cosmo_cache=cosmo_cache,
                metallicity_values=metallicity_values,
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
