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
    PREVIEW_SCALE = 0.3
    # ASSUME: all frames are identical in size. This is a rectangle that defines the size of a single full frame.
    full_frame = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))

    # Determine the nominal offsets based upon the machine's programmed x/y coordinates
    # for the image, based on the nominal stepping programmed into the imaging run.
    # For a perfect mechanical system:
    # moving_img + stepping_vector_px "aligns with" ref_img
    stepping_vector_px = Point(
        ((ref_meta['x'] * 1000 + ref_t['offset'][0]) - moving_meta['x'] * 1000) * Schema.PIX_PER_UM,
        ((ref_meta['y'] * 1000 + ref_t['offset'][1]) - moving_meta['y'] * 1000) * Schema.PIX_PER_UM
    )
    # create an initial "template" based on the region of overlap between the reference and moving images
    template_rect_full = full_frame.intersection(full_frame.translate(stepping_vector_px))
    if template_rect_full is None:
        self.schema.store_auto_align_result(moving_layer, None, False)
        logging.warning("No overlap between reference and moving frame")
        err = np.zeros((600, 1000), dtype=np.uint8)
        cv2.putText(
            err, 'NO OVERLAP',
            org=(100, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )
        cv2.imshow('before/after', err)
        cv2.waitKey() # pause because no delay is specified
        return
    # scale down the intersection template so we have a search space:
    # It's a balance between search space (where you can slide the template around)
    # and specificity (bigger template will have a chance of encapsulating more features)
    SEARCH_SCALE = 0.65
    template_rect = template_rect_full.scale(SEARCH_SCALE)
    template = moving_img[
        int(template_rect.tl.y) : int(template_rect.br.y),
        int(template_rect.tl.x) : int(template_rect.br.x)
    ].copy()
    # trace the template's extraction point back to the moving rectangle's origin
    scale_offset = template_rect.tl - template_rect_full.tl
    template_ref = (-stepping_vector_px).clamp_zero() + scale_offset

    # stash a copy of the "before" stitch image for UI purposes
    ref_initial = template_rect.translate(-template_rect.tl).translate(template_ref)
    before = np.hstack((
        cv2.resize(template, None, None, PREVIEW_SCALE, PREVIEW_SCALE),
        cv2.resize(ref_img[
            int(ref_initial.tl.y):int(ref_initial.br.y),
            int(ref_initial.tl.x):int(ref_initial.br.x)
        ], None, None, PREVIEW_SCALE, PREVIEW_SCALE)
    ))

    # for performance benchmarking
    from datetime import datetime
    start = datetime.now()

    # normalize, and take the laplacian so we're looking mostly at edges and not global lighting gradients
    ref_norm = cv2.normalize(ref_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    template_norm = cv2.normalize(template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
    template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

    # apply template matching. If the ref image and moving image are "perfectly aligned",
    # the value of `match_pt` should be equal to `template_ref`
    # i.e. alignment criteria: match_pt - template_ref = (0, 0)
    if True:
        METHOD = cv2.TM_CCOEFF  # convolutional matching
        res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        match_pt = max_loc
        res_8u = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ret, thresh = cv2.threshold(res_8u, 224, 255, 0)
    else:
        METHOD = cv2.TM_SQDIFF  # squared error matching - not as good as convolutional matching for our purposes
        res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        match_pt = min_loc
        res_8u = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        res_8u = 255 - res_8u # invert the thresholding
        ret, thresh = cv2.threshold(res_8u, 224, 255, 0)

    # find contours of candidate matches
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res_8u, contours, -1, (0,255,0), 1)
    cv2.imshow('contours', res_8u)

    # use the contours and the matched point to measure the quality of the template match
    has_single_solution = True
    score = None
    for index, c in enumerate(contours):
        if hierarchy[0][index][3] == -1:
            if cv2.pointPolygonTest(c, match_pt, False) > 0: # detect if point is inside the contour
                if score is not None:
                    has_single_solution = False
                score = cv2.contourArea(c)
                logging.debug(f"countour {c} contains {match_pt} and has area {score}")
                logging.info(f"                    score: {score}")
            else:
                # print(f"countour {c} does not contain {top_left}")
                pass
        else:
            if cv2.pointPolygonTest(c, match_pt, False) > 0:
                logging.info(f"{match_pt} is contained within a donut-shaped region. Suspect blurring error!")
                has_single_solution = False

    if score is not None and has_single_solution: # store the stitch if a good match was found
        adjustment_vector_px = Point(
            match_pt[0] - template_ref[0],
            match_pt[1] - template_ref[1]
        )
        logging.debug("template search done in {}".format(datetime.now() - start))
        logging.debug(f"minima at: {match_pt}")
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
            not has_single_solution,
        )
        check_t = self.schema.schema['tiles'][str(moving_layer)]
        logging.info(f"after adjustment: {check_t['offset'][0]:0.2f},{check_t['offset'][1]:0.2f}")

        # assemble the before/after images
        after_vector_px = Point(
            (ref_meta['x'] * 1000 + ref_t['offset'][0] - (moving_meta['x'] * 1000 + check_t['offset'][0])) * Schema.PIX_PER_UM,
            (ref_meta['y'] * 1000 + ref_t['offset'][1] - (moving_meta['y'] * 1000 + check_t['offset'][1])) * Schema.PIX_PER_UM
        )
        ref_overlap = full_frame.intersection(
            full_frame.translate(Point(0, 0) - after_vector_px)
        )
        moving_overlap = full_frame.intersection(
            full_frame.translate(after_vector_px)
        )
        after = np.hstack(pad_images_to_same_size(
            (
                cv2.resize(moving_img[
                    int(moving_overlap.tl.y):int(moving_overlap.br.y),
                    int(moving_overlap.tl.x):int(moving_overlap.br.x)
                ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE),
                cv2.resize(ref_img[
                    int(ref_overlap.tl.y) : int(ref_overlap.br.y),
                    int(ref_overlap.tl.x) : int(ref_overlap.br.x)
                ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE)
            )
        ))
        overview = np.hstack(pad_images_to_same_size(
            (
                cv2.resize(moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                cv2.resize(ref_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2)
            )
        ))
        before_after = np.vstack(
            pad_images_to_same_size(
                (before, after, overview)
            )
        )
        cv2.imshow('before/after', before_after)
        cv2.waitKey(10)
    else: # pause on errors
        # store the error, as score = None
        self.schema.store_auto_align_result(
            moving_layer,
            score,
            not has_single_solution,
        )
        logging.warning("No alignment found")
        after = np.zeros(before.shape, dtype=np.uint8)
        cv2.putText(
            after, "AUTOALIGN FAIL",
            org=(100, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )
        before_after = np.vstack(
            pad_images_to_same_size(
                (before, after)
            )
        )
        cv2.imshow('before/after', before_after)
        cv2.waitKey() # pause because no delay is specified



def stitch_auto_template(self):
    STRIDE_X_MM = Schema.NOM_STEP
    STRIDE_Y_MM = Schema.NOM_STEP

    extents = self.schema.br_centroid
    if False: # Set to just stitch the top two lines for faster debugging
        extents[1] = self.schema.tl_centroid[1] + Schema.NOM_STEP * 2
    # find an anchor layer
    # start from the smallest coordinates in x/y and work our way up along X, then along Y.
    found_anchor = False
    x_roll = 0
    y_roll = 0
    y_indices = np.arange(self.schema.tl_centroid[1], extents[1] + STRIDE_Y_MM, STRIDE_Y_MM)
    x_indices = np.arange(self.schema.tl_centroid[0], extents[0] + STRIDE_X_MM, STRIDE_X_MM)
    y_steps = 0
    for y in y_indices:
        x_steps = 0
        for x in x_indices:
            (ref_layer, ref_t) = self.schema.get_tile_by_coordinate(Point(x, y))
            if ref_t is not None:
                if ref_t['auto_error'] == 'anchor':
                    ref_img = self.schema.get_image_from_tile(ref_t)
                    ref_t['score'] = 1.0 # made up number that is positive to indicate we've been placed
                    ref_t['auto_error'] = 'anchor'
                    found_anchor = True
                    x_roll = x_steps
                    y_roll = y_steps
            x_steps += 1
        y_steps += 1
    if not found_anchor:
        logging.error("No anchor layer set, can't proceed with stitching")
        return
    y_indices = np.roll(y_indices, -y_roll) # "roll" the indices so we start at the anchor
    # take the last indices after the roll and invert their order, so we're always working backwards from the anchor
    # [-y_roll:] -> take the indices from the end to end - y_roll
    # [::-1] -> invert order
    # [:-y_roll] -> take the indices from the beginning to the end - y_roll
    y_indices = np.concatenate([y_indices[:-y_roll], y_indices[-y_roll:][::-1]])
    x_indices = np.roll(x_indices, -x_roll)
    x_indices = np.concatenate([x_indices[:-x_roll], x_indices[-x_roll:][::-1]])

    # start stitching from the anchor point
    for y in y_indices:
        for x in x_indices:
            overlaps = self.schema.get_intersecting_tiles(Point(x, y))
            combs = list(combinations(overlaps, 2))
            logging.debug(f"number of combinations @ ({x:0.2f}, {y:0.2f}): {len(combs)}")
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
                logging.info(f"overlap: {candidates[0].overlap_area:0.4f}")
                ref_img = self.schema.get_image_from_tile(ref_t)
                (moving_r, moving_layer, moving_t) = candidates[0].get_moving()
                moving_img = self.schema.get_image_from_tile(moving_t)
                logging.info(f"Stitching {ref_layer}:{ref_r.center()} to {moving_layer}:{moving_r.center()}")
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
