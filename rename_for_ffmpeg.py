#! /usr/bin/env python3

import argparse
from pathlib import Path
import shutil

def meta_from_fname(fname):
    metadata = {}
    items = fname.split('_')
    for i in items:
        metadata[i[0]] = float(i[1:])
    return metadata

def fname_from_meta(meta):
    fname = ''
    for k,v in meta.items():
        if isinstance(v, (float, int)):
            if fname != '':
                fname += '_'
            fname += str(k)
            if k == 'x' or k == 'y' or k == 'z':
                fname += f"{v:.2f}"
            elif k == 'a':
                fname += f"{int(v)}"
            else:
                fname += f"{int(v)}"
        else:
            # ignore other objects that were made, like the 'r_um' Rect
            pass
    return fname

def main():
    parser = argparse.ArgumentParser(description="Rename files for ffmpeg")
    parser.add_argument(
        "--idir", required=False, help="input directory", default="."
    )
    parser.add_argument(
        "--oname", required=False, help="base name of outputs", default='-ffmpeg.png'
    )
    parser.add_argument(
        "--odir", required=False, help="directory for outputs", default="anim"
    )
    args = parser.parse_args()

    base_path = Path(args.idir)
    files = [file for file in base_path.glob('*.png') if file.is_file()]

    # Load based on filenames, and finalize the overall area
    flist = {}
    for file in files:
        if '_r' + str(2) in file.stem: # filter image revs by the initial default rev
            meta = meta_from_fname(file.stem)
            flist[meta['a']] = meta

    flist = sorted(flist.items())

    for i, (_u, file) in enumerate(flist):
        shutil.copy('./' + fname_from_meta(file) + '.png', args.odir + '/' + str(i).zfill(4) + args.oname)

if __name__ == "__main__":
    main()
