from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging

# Use template matching of laplacians to do stitching
def stitch_one_template(self):
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

    if ref_img is None or moving_img is None:
        logging.warning("Couldn't find reference or moving image, aborting!")
        return

    nominal_vector_px = (
        (moving_meta['x'] - ref_meta['x'] + moving_t['offset'][0]) * 1000 * Schema.PIX_PER_UM,
        (moving_meta['y'] - ref_meta['y'] + moving_t['offset'][1]) * 1000 * Schema.PIX_PER_UM
    )
    if nominal_vector_px[0] >= 0:
        intersected_x = (0, Schema.X_RES - nominal_vector_px[0])
    else:
        intersected_x = (0 - nominal_vector_px[0], Schema.X_RES)
    if nominal_vector_px[1] >= 0:
        intersected_y = (0, Schema.Y_RES - nominal_vector_px[1])
    else:
        intersected_y = (0 - nominal_vector_px[1], Schema.Y_RES)
    intersected_rect = Rect(
        Point(intersected_x[0], intersected_y[0]),
        Point(intersected_x[1], intersected_y[1])
    )
    intersected_rect = intersected_rect.scale(0.65)
    template = moving_img[
        int(intersected_rect.tl.y) : int(intersected_rect.br.y),
        int(intersected_rect.tl.x) : int(intersected_rect.br.x)
    ]
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    from datetime import datetime
    start = datetime.now()
    ref_norm = cv2.normalize(ref_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    template_norm = cv2.normalize(template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
    template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

    # Apply template Matching
    if True:
        METHOD = cv2.TM_CCOEFF  # convolutional matching
        res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
    else:
        METHOD = cv2.TM_SQDIFF  # squared error matching
        res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(ref_img, top_left, bottom_right, 255, 2)
    SCALE = 0.5
    # cv2.imshow('template', cv2.resize(template, None, None, SCALE, SCALE))
    cv2.imshow('matching result', cv2.resize(cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), None, None, SCALE, SCALE))
    # cv2.imshow('detected point', cv2.resize(ref_img, None, None, SCALE, SCALE))
    # cv2.imshow('moving image', cv2.resize(moving_img, None, None, SCALE, SCALE))
    # cv2.waitKey()

    adjustment_vector_px = Point(
        -nominal_vector_px[0] - (intersected_rect.tl.x - top_left[0]),
        -nominal_vector_px[1] - (intersected_rect.tl.y - top_left[1])
    )
    print("template search done in {}".format(datetime.now() - start))
    print(f"minima at: {top_left}")
    print(f"before adjustment: {moving_t['offset'][0]},{moving_t['offset'][1]}")
    # now update the offsets to reflect this
    self.schema.adjust_offset(
        self.selected_layer,
        adjustment_vector_px.x / Schema.PIX_PER_UM,
        adjustment_vector_px.y / Schema.PIX_PER_UM
    )
    check_t = self.schema.schema['tiles'][str(self.selected_layer)]
    print(f"after adjustment: {check_t['offset'][0]},{check_t['offset'][1]}")


