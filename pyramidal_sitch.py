from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging

# TODO recode this template
# using pyramidal downsampling
# src = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))

def stitch_one_pyramidal(self):
    # algorithm:
    #
    # Generate pyramidal decompositions of both source and destinations
    # At the target height, do an exhaustive search for the best-fit MSE coordinate
    # Then descend the pyramid, re-doing the search at each level, seeding at the previous level's coordinate

    ref_img = None
    moving_img = None
    # extract the reference tile and moving tile
    for (layer, t, img) in self.schema.zoom_cache:
        meta = Schema.meta_from_tile(t)
        if layer == self.ref_layer:
            ref_img = img
            ref_meta = meta
            ref_t = t
        elif layer == self.selected_layer:
            moving_img = img
            moving_meta = meta
            moving_t = t

    nominal_vector_px = (
        (moving_meta['x'] - ref_meta['x']) * 1000 * Schema.PIX_PER_UM,
        (moving_meta['y'] - ref_meta['y']) * 1000 * Schema.PIX_PER_UM
    )
    if nominal_vector_px[0] >= 0:
        intersected_x = (0, Schema.X_RES - nominal_vector_px[0])
    else:
        intersected_x = (Schema.X_RES + nominal_vector_px[0], Schema.X_RES)
    if nominal_vector_px[1] >= 0:
        intersected_y = (0, Schema.Y_RES - nominal_vector_px[1])
    else:
        intersected_y = (Schema.Y_RES + nominal_vector_px[1], Schema.Y_RES)
    intersected_rect = Rect(
        Point(intersected_x[0], intersected_y[0]),
        Point(intersected_x[1], intersected_y[1])
    )
    intersected_rect = intersected_rect.scale(0.75)
    template = moving_img[
        int(intersected_rect.tl.y) : int(intersected_rect.br.y),
        int(intersected_rect.tl.x) : int(intersected_rect.br.x)
    ]
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = ref_img.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        cv2.imshow('template', template)
        cv2.imshow('matching result', res)
        cv2.imshow('detected point', img)
        cv2.imshow('moving image', moving_img)
        cv2.waitKey()

