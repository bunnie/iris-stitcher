from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging
from itertools import combinations
from utils import pad_images_to_same_size

# low scores are better. scores greater than this fail.
FAILING_SCORE = 50.0
# maximum number of potential solutions before falling back to manual review
MAX_SOLUTIONS = 16

# Use template matching of laplacians to do stitching
def stitch_one_template(self,
                        ref_img, ref_meta, ref_t,
                        moving_img, moving_meta, moving_t,
                        moving_layer):
    PREVIEW_SCALE = 0.3
    # ASSUME: all frames are identical in size. This is a rectangle that defines the size of a single full frame.
    full_frame = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))

    # Use the reference's offset as an initial "seed guess" for the moving frame
    self.schema.adjust_offset(
        moving_layer,
        ref_t['offset'][0],
        ref_t['offset'][1]
    )

    # Determine the nominal offsets based upon the machine's programmed x/y coordinates
    # for the image, based on the nominal stepping programmed into the imaging run.
    # For a perfect mechanical system:
    # moving_img + stepping_vector_px "aligns with" ref_img
    stepping_vector_px = Point(
        ((ref_meta['x'] * 1000 + ref_t['offset'][0])
            - (moving_meta['x'] * 1000 + moving_t['offset'][0])) * Schema.PIX_PER_UM,
        ((ref_meta['y'] * 1000 + ref_t['offset'][1])
            - (moving_meta['y'] * 1000 + moving_t['offset'][1])) * Schema.PIX_PER_UM
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
        overview = np.hstack(pad_images_to_same_size(
            (
                cv2.resize(moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                cv2.resize(ref_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2)
            )
        ))
        cv2.imshow('before/after', np.vstack(
            pad_images_to_same_size((err, overview))
        ))
        cv2.waitKey() # pause because no delay is specified
        return
    # scale down the intersection template so we have a search space:
    # It's a balance between search space (where you can slide the template around)
    # and specificity (bigger template will have a chance of encapsulating more features)
    SEARCH_SCALE = 0.8
    template_rect = template_rect_full.scale(SEARCH_SCALE)
    template = moving_img[
        round(template_rect.tl.y) : round(template_rect.br.y),
        round(template_rect.tl.x) : round(template_rect.br.x)
    ].copy()
    # trace the template's extraction point back to the moving rectangle's origin
    scale_offset = template_rect.tl - template_rect_full.tl
    template_ref = (-stepping_vector_px).clamp_zero() + scale_offset

    # stash a copy of the "before" stitch image for UI purposes
    ref_initial = template_rect.translate(-template_rect.tl).translate(template_ref)
    before = np.hstack(pad_images_to_same_size((
        cv2.resize(template, None, None, PREVIEW_SCALE, PREVIEW_SCALE),
        cv2.resize(ref_img[
            round(ref_initial.tl.y):round(ref_initial.br.y),
            round(ref_initial.tl.x):round(ref_initial.br.x)
        ], None, None, PREVIEW_SCALE, PREVIEW_SCALE)
    )))

    stitch_again = True
    manual_review = False
    while stitch_again:
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
            if hierarchy[0][index][3] == -1 and len(hierarchy[0]) < MAX_SOLUTIONS:
                if cv2.pointPolygonTest(c, match_pt, False) >= 0.0: # detect if point is inside or on the contour. On countour is necessary to detect cases of exact matches.
                    if score is not None:
                        has_single_solution = False
                    score = cv2.contourArea(c)
                    logging.debug(f"countour {c} contains {match_pt} and has area {score}")
                    logging.debug(f"                    score: {score}")
                else:
                    # print(f"countour {c} does not contain {top_left}")
                    pass
            else:
                if cv2.pointPolygonTest(c, match_pt, False) > 0:
                    logging.info(f"{match_pt} is contained within a donut-shaped region. Suspect blurring error!")
                    has_single_solution = False

        if score is not None and has_single_solution and not manual_review: # store the stitch if a good match was found
            while True:
                adjustment_vector_px = Point(
                    match_pt[0] - template_ref[0],
                    match_pt[1] - template_ref[1]
                )
                if score < FAILING_SCORE:
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
                    logging.info(f"after adjustment: {check_t['offset'][0]:0.2f}, {check_t['offset'][1]:0.2f} score: {score} candidates: {len(hierarchy[0])}")

                    # assemble the before/after images
                    after_vector_px = Point(
                        round((ref_meta['x'] * 1000 + ref_t['offset'][0]
                            - (moving_meta['x'] * 1000 + check_t['offset'][0])) * Schema.PIX_PER_UM),
                        round((ref_meta['y'] * 1000 + ref_t['offset'][1]
                            - (moving_meta['y'] * 1000 + check_t['offset'][1])) * Schema.PIX_PER_UM)
                    )
                    ref_overlap = full_frame.intersection(
                        full_frame.translate(Point(0, 0) - after_vector_px)
                    )
                    moving_overlap = full_frame.intersection(
                        full_frame.translate(after_vector_px)
                    )
                    if ref_overlap is None or moving_overlap is None:
                        logging.error("hard error: no overlap despite a passing score!")
                    after = np.hstack(pad_images_to_same_size(
                        (
                            cv2.resize(moving_img[
                                round(moving_overlap.tl.y):round(moving_overlap.br.y),
                                round(moving_overlap.tl.x):round(moving_overlap.br.x)
                            ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE),
                            cv2.resize(ref_img[
                                round(ref_overlap.tl.y) : round(ref_overlap.br.y),
                                round(ref_overlap.tl.x) : round(ref_overlap.br.x)
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
                    break
                else:
                    # compute after vector without storing the result
                    after_vector_px = Point(
                        round((ref_meta['x'] * 1000 + ref_t['offset'][0]
                            - (moving_meta['x'] * 1000 + moving_t['offset'][0] + adjustment_vector_px.x / Schema.PIX_PER_UM)) * Schema.PIX_PER_UM),
                        round((ref_meta['y'] * 1000 + ref_t['offset'][1]
                            - (moving_meta['y'] * 1000 + moving_t['offset'][1] + adjustment_vector_px.y / Schema.PIX_PER_UM)) * Schema.PIX_PER_UM)
                    )
                    ref_overlap = full_frame.intersection(
                        full_frame.translate(Point(0, 0) - after_vector_px)
                    )
                    moving_overlap = full_frame.intersection(
                        full_frame.translate(after_vector_px)
                    )
                    # display the difference of laplacians of the overlapping region
                    moving_roi = moving_img[
                        round(moving_overlap.tl.y):round(moving_overlap.br.y),
                        round(moving_overlap.tl.x):round(moving_overlap.br.x)
                    ]
                    ref_roi = ref_img[
                        round(ref_overlap.tl.y) : round(ref_overlap.br.y),
                        round(ref_overlap.tl.x) : round(ref_overlap.br.x)
                    ]
                    ref_norm = cv2.normalize(ref_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    moving_norm = cv2.normalize(moving_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
                    moving_laplacian = cv2.Laplacian(moving_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
                    corr = moving_laplacian - ref_laplacian
                    corr_u8 = cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cv2.imshow("manual alignment", corr_u8)

                    # get user feedback and adjust the match_pt accordingly
                    logging.warning(f"Stitch score {score} > {FAILING_SCORE}: wasd to move, 1 to accept, 2 to remove image")
                    key = cv2.waitKey()
                    if key != -1:
                        key_char = chr(key)
                        logging.debug(f'Got key: {key_char}')
                        if key_char == ',' or key_char == 'w':
                            match_pt = (match_pt[0] + 0, match_pt[1] - 1)
                        elif key_char == 'a':
                            match_pt = (match_pt[0] - 1, match_pt[1] + 0)
                        elif key_char == 'e' or key_char == 'd':
                            match_pt = (match_pt[0] + 1, match_pt[1] + 0)
                        elif key_char == 'o' or key_char == 's':
                            match_pt = (match_pt[0] + 0, match_pt[1] + 1)
                        elif key_char == '1': # this accepts the current alignment
                            score = FAILING_SCORE - 1
                        elif key_char == '2': # this rejects the alignment
                            self.schema.remove_tile(moving_layer)
                            logging.info(f"Removing tile {moving_layer} from the database")
                            break
                        else:
                            logging.debug(f"Unhandled key: {key_char}, ignoring")
            stitch_again = False # exit the main retry loop

        else: # pause on errors
            logging.warning(f"Stitch failure: single solution {has_single_solution}, score {score}")
            if score is not None: # score was generated, but failed
                logging.warning(f"Score of {score} > {FAILING_SCORE}, marking region as stitch failure")
                score = None
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
            overview = np.hstack(pad_images_to_same_size(
                (
                    cv2.resize(moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                    cv2.resize(ref_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2)
                )
            ))
            before_after = np.vstack(
                pad_images_to_same_size(
                    (cv2.resize(template, None, None, PREVIEW_SCALE, PREVIEW_SCALE),
                     after, overview)
                )
            )
            cv2.imshow('before/after', before_after)
            logging.info("press wasd to adjust template region, 1 to accept as-is, press 2 to remove image")
            key = cv2.waitKey() # pause because no delay is specified
            SHIFT_AMOUNT = 50
            if key != -1:
                key_char = chr(key)
                if key_char == ',' or key_char == 'w':
                    template_shift = Point(0, -SHIFT_AMOUNT)
                elif key_char == 'a':
                    template_shift = Point(-SHIFT_AMOUNT, 0)
                elif key_char == 'e' or key_char == 'd':
                    template_shift = Point(SHIFT_AMOUNT, 0)
                elif key_char == 'o' or key_char == 's':
                    template_shift = Point(0, SHIFT_AMOUNT)
                elif key_char == '2':
                    self.schema.remove_tile(moving_layer)
                    logging.info(f"Removing tile {moving_layer} from the database")
                    stitch_again = False
                    template_shift = None
                    manual_review = False # go back to autostitching
                elif key_char == '1':
                    logging.info("Accepting image as-is.")
                    stitch_again = False
                    template_shift = None
                    manual_review = False # go back to autostitching

                if template_shift is not None:
                    manual_review = True # check the result before accepting it
                    template_rect = template_rect.saturating_translate(template_shift, full_frame)
                    template = moving_img[
                        round(template_rect.tl.y) : round(template_rect.br.y),
                        round(template_rect.tl.x) : round(template_rect.br.x)
                    ].copy()
                    # trace the template's extraction point back to the moving rectangle's origin
                    scale_offset = template_rect.tl - template_rect_full.tl
                    template_ref = (-stepping_vector_px).clamp_zero() + scale_offset
                # this should wrap around to the top and try another template match


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
    if y_roll != 0:
        y_indices = np.roll(y_indices, -y_roll) # "roll" the indices so we start at the anchor
        # take the last indices after the roll and invert their order, so we're always working backwards from the anchor
        # [-y_roll:] -> take the indices from the end to end - y_roll
        # [::-1] -> invert order
        # [:-y_roll] -> take the indices from the beginning to the end - y_roll
        y_indices = np.concatenate([y_indices[:-y_roll], y_indices[-y_roll:][::-1]])
    if x_roll != 0:
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
                # filter by candidates only -- pairs that have one aligned, and one unaligned image.
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
                logging.debug(f"overlap: {candidates[0].overlap_area:0.4f}")
                ref_img = self.schema.get_image_from_tile(ref_t)
                (moving_r, moving_layer, moving_t) = candidates[0].get_moving()
                moving_img = self.schema.get_image_from_tile(moving_t)
                logging.info(f"Stitching {ref_layer}:{ref_r.center()} to {moving_layer}:{moving_r.center()}")
                self.stitch_one_template(
                    ref_img, Schema.meta_from_tile(ref_t), ref_t,
                    moving_img, Schema.meta_from_tile(moving_t), moving_t, moving_layer
                )
    logging.info("Auto-stitch pass done")

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
