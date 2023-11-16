#! /usr/bin/env python3

import argparse
from pathlib import Path
from schema import Schema
import numpy as np
import shutil

# x:1067px = 0.453mm / x:1180 = 0.5017mm  y:1443px = 0.613mm / y:1400px = 0.595mm
X_MACH_SCALE = 1.000
Y_MACH_SCALE = 1.190

def main():
    parser = argparse.ArgumentParser(description="IRIS Stitching Scripts")
    parser.add_argument(
        "--src", required=True, help="source dir"
    )
    parser.add_argument(
        "--dst", required=True, help="destination dir"
    )
    parser.add_argument(
        "--dry-run", help="Do a dry run (don't copy any files)", default=False, action='store_true'
    )
    parser.add_argument(
        "--x", required=False, help="Rescale factor for x positions", default=1.0, type=float
    )
    parser.add_argument(
        "--y", required=False, help="Rescale factor for y positions", default=1.0, type=float
    )
    args = parser.parse_args()

    src_path = Path("raw/" + args.src)
    src_files = [file for file in src_path.glob('*.png') if file.is_file()]

    coords_mm = []
    for file in src_files:
        metadata = Schema.meta_from_fname(file.stem)
        coords_mm += [(metadata['x'], metadata['y'])]

    coords = np.unique(coords_mm, axis=0)
    dists = []
    for p in coords:
        dists += [np.linalg.norm(p - [-100, -100])]
    ll_centroid = coords[dists.index(min(dists))]

    # ll_centroid is our "reference point". From here, we scale everything else.
    for file in src_files:
        metadata = Schema.meta_from_fname(file.stem)
        metadata['x'] = (float(metadata['x']) - ll_centroid[0]) * args.x + ll_centroid[0]
        metadata['y'] = (float(metadata['y']) - ll_centroid[1]) * args.y + ll_centroid[1]
        new_fname = Schema.fname_from_meta(metadata)
        if args.dry_run:
            print(f"{file.stem} -> {new_fname}")
        else:
            dest_path = Path(f"raw/{args.dst}") / (new_fname + ".png")
            print(f"copying {file} -> {dest_path}")
            shutil.copy(file, dest_path)

if __name__ == "__main__":
    main()
