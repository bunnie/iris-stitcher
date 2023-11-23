from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging

# Use template matching of laplacians to do stitching
def stitch_one_template(self, ref_img, ref_meta, ref_t, moving_img, moving_meta, moving_t, moving_layer):
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
    # cv2.imshow('matching result', cv2.resize(cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX), None, None, SCALE, SCALE))
    # cv2.imshow('matching result', cv2.normalize(res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX))
    res_8u = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imshow('matching result', res_8u)
    ret, thresh = cv2.threshold(res_8u, 192, 255, 0)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res_8u, contours, -1, (0,255,0), 1)
    cv2.imshow('contours', res_8u)
    cv2.waitKey()
    has_single_solution = True
    score = None
    for index, c in enumerate(contours):
        if hierarchy[0][index][3] == -1:
            if cv2.pointPolygonTest(c, top_left, False) > 0: # detect if point is inside the contour
                if score is not None:
                    has_single_solution = False
                score = cv2.contourArea(c)
                logging.debug(f"countour {c} contains {top_left} and has area {score}")
            else:
                # print(f"countour {c} does not contain {top_left}")
                pass
        else:
            if cv2.pointPolygonTest(c, top_left, False) > 0:
                logging.debug(f"{top_left} is contained within a donut-shaped region. Suspect blurring error!")
                has_single_solution = False

    # cv2.imshow('detected point', cv2.resize(ref_img, None, None, SCALE, SCALE))
    # cv2.imshow('moving image', cv2.resize(moving_img, None, None, SCALE, SCALE))
    # cv2.waitKey()

    adjustment_vector_px = Point(
        -nominal_vector_px[0] - (intersected_rect.tl.x - top_left[0]),
        -nominal_vector_px[1] - (intersected_rect.tl.y - top_left[1])
    )
    logging.debug("template search done in {}".format(datetime.now() - start))
    logging.debug(f"minima at: {top_left}")
    logging.debug(f"before adjustment: {moving_t['offset'][0]},{moving_t['offset'][1]}")
    # now update the offsets to reflect this
    self.schema.adjust_offset(
        moving_layer,
        adjustment_vector_px.x / Schema.PIX_PER_UM,
        adjustment_vector_px.y / Schema.PIX_PER_UM
    )
    self.schema.store_auto_align_result(
        moving_layer,
        score,
        has_single_solution,
    )
    check_t = self.schema.schema['tiles'][str(moving_layer)]
    logging.info(f"after adjustment: {check_t['offset'][0]},{check_t['offset'][1]}")


def stitch_auto_template(self):
    STRIDE_X_MM = 0.1
    STRIDE_Y_MM = 0.1

    # start from the smallest coordinates in x/y and work our way up along X, then along Y.
    (ref_layer, ref_t) = self.schema.get_tile_by_coordinate(self.schema.tl_centroid)
    assert ref_layer is not None, "Couldn't find initial tile!"
    ref_img = self.schema.get_image_from_tile(ref_t)

    extents = self.schema.br_centroid

    for y in np.arange(self.schema.tl_centroid[1], extents[1], STRIDE_Y_MM):
        if ref_layer is not None:
            for x in np.arange(self.schema.tl_centroid[0], extents[0], STRIDE_X_MM):
                logging.info(f"{x}, {y}")
                (moving_layer, moving_t) = self.schema.get_tile_by_coordinate(Point(x, y))
                if moving_layer == ref_layer or moving_layer is None:
                    continue
                else:
                    logging.info(f"Trying to stitch {ref_layer} and {moving_layer}")
                    moving_img = self.schema.get_image_from_tile(moving_t)
                    self.stitch_one_template(
                        ref_img, Schema.meta_from_tile(ref_t), ref_t,
                        moving_img, Schema.meta_from_tile(moving_t), moving_t, moving_layer
                    )
                    ref_layer = moving_layer
            last_ref = ref_layer
            ref_layer = None
        else:
            (candidate_layer, candidate_t) = self.schema.get_tile_by_coordinate(Point(x, y))
            if candidate_layer == last_ref or candidate_layer is None:
                continue
            else:
                ref_layer = candidate_layer
                ref_t = candidate_t
                ref_img = self.schema.get_image_from_tile(ref_t)


