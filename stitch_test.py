#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2
import sys

from math import log2, ceil

def make_gaussian_pyramid(base, levels):
    g = base.copy()
    pyramid = [g]
    for i in range(levels):
        g = cv2.pyrDown(g)
        pyramid += [g]
    return pyramid

def make_laplacian_from_gaussian(gaussian):
    lp = [gaussian[-1]]
    for i in range(len(gaussian) - 1, 0, -1):
        ge = cv2.pyrUp(gaussian[i])
        l = cv2.subtract(gaussian[i-1], ge)
        lp += [l]
    return lp

def square_image(img, pad=0):
    # square up an image to the nearest power of 2
    max_dim = max(img.shape[0], img.shape[1])
    max_dim = 2**ceil(log2(max_dim))
    sq_canvas = np.full((max_dim, max_dim), pad, dtype=np.uint8)
    # Calculate the position to paste the non-square image in the center
    x_offset = (sq_canvas.shape[1] - img.shape[1]) // 2
    y_offset = (sq_canvas.shape[0] - img.shape[0]) // 2

    # Paste the non-square image in the center of the square canvas
    sq_canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    return sq_canvas, (x_offset, y_offset)

def composite_gaussian_pyramid(pyramid):
    rows, cols = pyramid[0].shape
    # determine the total number of rows and columns for the composite
    composite_rows = max(rows, sum(p.shape[0] for p in pyramid[1:]))
    composite_cols = cols + pyramid[1].shape[1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.uint8)

    # store the original to the left
    composite_image[:rows, :cols] = pyramid[0]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    return composite_image

def composite_laplacian_pyramid(pyramid):
    rows, cols = pyramid[-1].shape
    # determine the total number of rows and columns for the composite
    composite_rows = max(rows, sum(p.shape[0] for p in pyramid[:-1]))
    composite_cols = cols + pyramid[-2].shape[1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.uint8)

    # store the original to the left
    composite_image[:rows, :cols] = pyramid[-1]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in reversed(pyramid[:-1]):
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    return composite_image


# https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded


def main():
    parser = argparse.ArgumentParser(description="IRIS Stitching Scripts")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=False, help="base name", default='test'
    )
    parser.add_argument(
        '--blend_strength', action='store', default=5,
        help="Blending strength from [0,100] range. The default is 5",
        type=np.int32, dest='blend_strength'
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    test_path = Path("./")
    files = [file for file in test_path.glob(f'{args.name}*.png') if file.is_file()]

    db = []
    for file in files:
        (_root, x, y, index) = file.stem.split('_')
        img = cv2.imread(str(test_path / file), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        db += [(img, int(x), int(y), int(index))]

    SCALE = 0.5
    CANVAS_W = 7000
    CANVAS_H = 7000

    corners = []
    images = []
    masks = []
    for (img, x, y, _index) in db:
        # the corner is the top left corner of where the image should go after alignment
        corners += [(x, y)]
        if False:
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
            canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
            images += [canvas]
            mask[y : y + img.shape[0], x : x + img.shape[1]] = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
            masks += [mask]
        else:
            images += [img]
            # the mask is 255 where pixels should be copied into the final mosaic canvas
            mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
            masks += [mask]

    # this computes the full size of the resulting canvas
    dst_sz = cv2.detail.resultRoi(corners=corners, images=images)

    # set up the blender algorithm. This case uses the Burt & Adelson 1983 multiresolution
    # spline algorithm (gaussian/laplacian pyramids) with some modern refinements that
    # haven't been explicitly documented by opencv.
    blender = cv2.detail_MultiBandBlender(try_gpu=1)
    # I *think* this sets how far the blending seam should go from the edge.
    blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * args.blend_strength / 100
    # I read "bands" as basically how deep you want the pyramids to go
    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
    # Allocates memory for the final image
    blender.prepare(dst_sz)

    # Feed the images into the blender itself
    for (img, mask, corner) in zip(images, masks, corners):
        print(corner)
        blender.feed(img, mask, corner)

    # The actual computational step.
    result, result_mask = blender.blend(None, None)

    # Show results
    cv2.imshow("blend",
        cv2.resize(cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U), None, None, SCALE, SCALE),
    )
    cv2.imshow("mask",
        cv2.resize(result_mask, None, None, SCALE, SCALE),
    )
    cv2.waitKey()

if __name__ == "__main__":
    main()
