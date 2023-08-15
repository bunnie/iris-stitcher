import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2

# derived from reference image "full-H"
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

def get_image(files, coord, r):
    img_re = re.compile('x([0-9.\-]*)_y([0-9.\-]*)_.*_r([\d*])')
    for file in files:
        match = img_re.match(file.stem).groups()
        if len(match) == 3:
            if coord[0] == float(match[0]) and coord[1] == float(match[1]) and r == int(match[2]):
                print(str(file))
                return cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    logging.error(f"Requested file was not found at {coord}, r={r}")
    return None

def main():
    parser = argparse.ArgumentParser(description="IRIS Stitching Scripts")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=False, help="name of image directory containing raw files", default='338s1285-b'
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    raw_image_path = Path("raw/" + args.name)
    files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

    # Coordinate system of images X/Y:
    # Y
    # ^
    # |
    # (0,0) ----> X
    #
    # Coordinate system of OpenCV and X/Y:
    # (0,0) ----> X
    # |
    # v
    # Y

    centroids = []
    for file in files:
        elems = file.stem.split('_')
        x = None
        y = None
        for e in elems:
            if 'x' in e:
                x = float(e[1:])
            if 'y' in e:
                y = float(e[1:])
        if (x is not None and y is None) or (y is not None and x is None):
            logging.error(f"only one coordinate found in {file.stem}")
        else:
            if x is not None and y is not None:
                centroids += [[x, y]]

    coords = np.unique(np.array(centroids), axis=0)

    # Find the "lower left" corner. This is done by computing the euclidian distance
    # from all the points to a point at "very lower left", i.e. -100, -100
    dists = []
    for p in coords:
        dists += [np.linalg.norm(p - [-100, -100])]
    ll = coords[dists.index(min(dists))]
    ur = coords[dists.index(max(dists))]
    print(f"Lower-left coordinate: {ll}; upper-right coordinate: {ur}")

    # Determine total area of imaging centroid
    x_mm_centroid = ur[0] - ll[0]
    y_mm_centroid = ur[1] - ll[1]
    # Determine absolute imaging area in pixels based on pixels/mm and image size
    # X_RES, Y_RES added because we have a total of one frame size surrounding the centroid
    x_res = int(math.ceil(x_mm_centroid * 1000 * PIX_PER_UM + X_RES))
    y_res = int(math.ceil(y_mm_centroid * 1000 * PIX_PER_UM + Y_RES))
    print(f"Final image resolution is {x_res}x{y_res}")

    canvas = np.zeros((y_res, x_res), dtype=np.uint8)

    # create a list of x-coordinates
    x_list = np.unique(coords[:, 0])

    # starting point for tiling into CV image space
    cv_y = 0
    cv_x = 0
    last_coord = None
    y_was_reset = False
    # now step along each x-coordinate and fetch the y-images
    for x in x_list:
        col_coords_swapped = np.sort(coords[coords[:, 0] == x])
        # the sort above ... swaps our x/y coordinates? ugh. Maybe something else subtle going on.
        col_coords = []
        for c in col_coords_swapped:
            col_coords += [[c[1], c[0]]]
        col_coords = np.array(col_coords)
        # now operate on the column list
        if last_coord is not None:
            delta_x_mm = abs(col_coords[0][0] - last_coord[0])
            delta_x_pix = int(delta_x_mm * 1000 * PIX_PER_UM)
            print(f"Stepping X by {delta_x_mm}mm -> {delta_x_pix}px")
            cv_x += delta_x_pix

            # restart the coordinate to the top
            cv_y = 0
            print(f"Resetting y coord to {cv_y}")
            y_was_reset = True
        for c in col_coords:
            img = get_image(files, c, r=2)
            if not y_was_reset and last_coord is not None:
                delta_y_mm = abs(c[1] - last_coord[1])
                delta_y_pix = int(delta_y_mm * 1000 * PIX_PER_UM)
                print(f"Stepping Y by {delta_y_mm}mm -> {delta_y_pix}px")
                cv_y += delta_y_pix
            else:
                y_was_reset = False

            # copy the image to the appointed region
            dest = canvas[cv_y: cv_y + Y_RES, cv_x:cv_x + X_RES]
            # Note to self: use `last_coord is None` as a marker if we should stitch or not
            # i.e., first image in the series or not.
            cv2.addWeighted(dest, 0, img, 1, 0, dest)

            last_coord = c

    cv2.imwrite('test2.png', canvas)

if __name__ == "__main__":
    main()
