from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging
from itertools import combinations

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

# Use template matching of laplacians to do stitching
def stitch_one_template(self,
                        ref_img, ref_meta, ref_t,
                        moving_img, moving_meta, moving_t,
                        moving_layer):
    nominal_vector_px = Point(
        ((moving_meta['x'] - ref_meta['x']) * 1000 + moving_t['offset'][0]) * Schema.PIX_PER_UM,
        ((moving_meta['y'] - ref_meta['y']) * 1000 + moving_t['offset'][1]) * Schema.PIX_PER_UM
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
    ].copy()
    ref_initial = intersected_rect.translate(nominal_vector_px)
    before = np.hstack((
        cv2.resize(template, None, None, 0.3, 0.3),
        cv2.resize(ref_img[
            int(ref_initial.tl.y):int(ref_initial.br.y),
            int(ref_initial.tl.x):int(ref_initial.br.x)
        ], None, None, 0.3, 0.3)
    ))

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
    has_single_solution = True
    score = None
    for index, c in enumerate(contours):
        if hierarchy[0][index][3] == -1:
            if cv2.pointPolygonTest(c, top_left, False) > 0: # detect if point is inside the contour
                if score is not None:
                    has_single_solution = False
                score = cv2.contourArea(c)
                logging.debug(f"countour {c} contains {top_left} and has area {score}")
                logging.info(f"                    score: {score}")
            else:
                # print(f"countour {c} does not contain {top_left}")
                pass
        else:
            if cv2.pointPolygonTest(c, top_left, False) > 0:
                logging.info(f"{top_left} is contained within a donut-shaped region. Suspect blurring error!")
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
    logging.info(f"after adjustment: {check_t['offset'][0]:0.2f},{check_t['offset'][1]:0.2f}")

    after_vector_px = Point(
        ((moving_meta['x'] - ref_meta['x']) * 1000 + moving_t['offset'][0]) * Schema.PIX_PER_UM,
        ((moving_meta['y'] - ref_meta['y']) * 1000 + moving_t['offset'][1]) * Schema.PIX_PER_UM
    )
    ref_after = intersected_rect.translate(after_vector_px)
    after = np.hstack(pad_images_to_same_size(
        (
            cv2.resize(template, None, None, 0.3, 0.3),
            cv2.resize(ref_img[
                int(ref_after.tl.y):int(ref_after.br.y),
                int(ref_after.tl.x):int(ref_after.br.x)
            ], None, None, 0.3, 0.3)
        )
    ))
    before_after = np.vstack(
        pad_images_to_same_size(
            (before, after)
        )
    )
    cv2.imshow('before/after', before_after)
    cv2.waitKey(10)


def stitch_auto_template(self):
    STRIDE_X_MM = Schema.NOM_STEP
    STRIDE_Y_MM = Schema.NOM_STEP

    # start from the smallest coordinates in x/y and work our way up along X, then along Y.
    (ref_layer, ref_t) = self.schema.get_tile_by_coordinate(self.schema.tl_centroid)
    assert ref_layer is not None, "Couldn't find initial tile!"
    ref_img = self.schema.get_image_from_tile(ref_t)
    ref_t['score'] = 1.0 # made up number that is positive to indicate we've been placed
    ref_t['auto_error'] = 'false'

    extents = self.schema.br_centroid

    for y in np.arange(self.schema.tl_centroid[1], extents[1], STRIDE_Y_MM):
        for x in np.arange(self.schema.tl_centroid[0], extents[0], STRIDE_X_MM):
            overlaps = self.schema.get_intersecting_tiles(Point(x, y))
            combs = list(combinations(overlaps, 2))
            print(f"number of combination @ ({x:0.2f}, {y:0.2f}): {len(combs)}")
            pairs = []
            for ((layer, t), (layer_o, t_o)) in combs:
                # filter by candidates only -- pairs that have one aligned, and one aligned image.
                if (t['score'] < 0.0 and t_o['score'] > 0.0) or (t['score'] > 0.0 and t_o['score'] < 0.0):
                    layer_meta = Schema.meta_from_tile(t)
                    r1 = Schema.rect_mm_from_center(Point(layer_meta['x'], layer_meta['y']))
                    layer_o_meta = Schema.meta_from_tile(t_o)
                    r2 = Schema.rect_mm_from_center(Point(layer_o_meta['x'], layer_o_meta['y']))
                    pairs += [
                        RectPair(r1, layer, t, r2, layer_o, t_o)
                    ]

            if len(pairs) > 0:
                logging.debug(f"{x:0.2f}, {y:0.2f}")
                candidates = sorted(pairs, reverse = True)
                (ref_r, ref_layer, ref_t) = candidates[0].get_ref()
                logging.info(f"overlap: {candidates[0].overlap_area}")
                ref_img = self.schema.get_image_from_tile(ref_t)
                (moving_r, moving_layer, moving_t) = candidates[0].get_moving()
                moving_img = self.schema.get_image_from_tile(moving_t)
                logging.debug(f"Stitching {ref_layer}:{ref_r} to {moving_layer}:{moving_r}")
                self.stitch_one_template(
                    ref_img, Schema.meta_from_tile(ref_t), ref_t,
                    moving_img, Schema.meta_from_tile(moving_t), moving_t, moving_layer
                )

class RectPair():
    def __init__(self, r1: Rect, r1_layer, r1_t, r2: Rect, r2_layer, r2_t):
        self.r1 = r1
        self.r2 = r2
        self.r1_layer = r1_layer
        self.r2_layer = r2_layer
        self.r1_t = r1_t
        self.r2_t = r2_t
        self.overlap = self.r1.intersection(self.r2)
        if self.overlap is not None:
            self.overlap_area = self.overlap.area()
        else:
            self.overlap_area = 0.0

    def __lt__(self, other):
        return self.overlap_area < other.overlap_area

    def __eq__(self, other):
        return round(self.overlap_area, 5) == round(other.overlap_area, 5)

    def is_dupe(self, rp):
        return (rp.r1 == self.r1 and rp.r2 == self.r2) or (rp.r1 == self.r2 and rp.r2 == self.r1)

    # True if exactly one of the pair has been scored, and the other has not (basically an XOR function).
    def is_candidate(self):
        return (self.r1_t['score'] < 0.0 and self.r2_t['score'] > 0.0) or (self.r1_t['score'] > 0.0 and self.r2_t['score'] < 0.0)

    def get_ref(self):
        if self.r1_t['score'] < 0.0 and self.r2_t['score'] > 0.0:
            return (self.r2, self.r2_layer, self.r2_t)
        elif self.r2_t['score'] < 0.0 and self.r1_t['score'] > 0.0:
            return (self.r1, self.r1_layer, self.r1_t)
        else:
            return None

    def get_moving(self):
        if self.r1_t['score'] < 0.0 and self.r2_t['score'] > 0.0:
            return (self.r1, self.r1_layer, self.r1_t)
        elif self.r2_t['score'] < 0.0 and self.r1_t['score'] > 0.0:
            return (self.r2, self.r2_layer, self.r2_t)
        else:
            return None
