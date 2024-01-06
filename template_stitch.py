from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging
from itertools import combinations
from utils import pad_images_to_same_size
from math import log10, ceil

# SOME TODO:
#  - seems like the match_pt result from template stitching has a coordinate inversion or synchronization
#    issue on one of the two option views
#  - the "Y" key that sets us to the default stitch -- is not correct for alternate views
#  - The left-right alternative doesn't seem to be immediately left-right. Is this because of
#    cum offset error, or messed up indexing? if cum offset error, maybe we need to correct for
#    that in picking an alternate...
#  - toggle off full_review and consider not doing schema overwrite once these are fixed...

# low scores are better. scores greater than this fail.
FAILING_SCORE = 100.0
# maximum number of potential solutions before falling back to manual review
MAX_SOLUTIONS = 12
PREVIEW_SCALE = 0.3
X_REVIEW_THRESH = 100.0
Y_REVIEW_THRESH = 100.0
SEARCH_SCALE = 0.80  # 0.8 worked on the AW set

class StitchState():

    def __init__(self, schema, ref_layers, moving_layer):
        self.schema = schema
        # base data
        self.ref_imgs = []
        self.ref_metas = []
        self.ref_tiles = []
        self.moving_img = None
        self.moving_meta = None
        self.moving_tile = None

        for ref_layer in ref_layers:
            tile = self.schema.schema['tiles'][ref_layer]
            if tile is None:
                logging.warning(f"layer {ref_layer} does not exist, skipping...")
            self.ref_metas += [Schema.meta_from_tile(tile)]
            self.ref_imgs += [self.schema.get_image_from_tile(tile)]
            self.ref_tiles += [tile]

        tile = self.schema.schema['tiles'][moving_layer]
        assert tile is not None, f"The layer to be stitched {moving_layer} is missing!"
        self.moving_img = self.schema.get_image_from_tile(tile)
        self.moving_meta = Schema.meta_from_tile(tile)
        self.moving_tile = tile

        ### derived data
        self.num_refs = len(ref_layers)
        # the proposed offset for the moving image for a given ref layer
        self.stepping_vectors = [None] * self.num_refs
        self.templates = [None] * self.num_refs
        # the rectangle that defines the template, relative to the moving image origin
        self.template_rects = [None] * self.num_refs
        # the full region of intersection between the ref & moving images
        self.intersection_rects = [None] * self.num_refs
        # the backtrack vector for the template - the vector needed to reverse the match point into an image offset
        self.template_backtrack = [None] * self.num_refs # was named 'template_refs'
        self.ref_laplacians = [None] * self.num_refs
        # the contours themselves
        self.contours = [None] * self.num_refs
        # hierarchy of contours that match (to discover nested solutions, etc.)
        self.hierarchies = [None] * self.num_refs
        # best match based on the current template convolution
        self.match_pts = [None] * self.num_refs
        # template convolved with reference image
        self.convolutions = [None] * self.num_refs # was named 'results'
        # tuple of (has_single_solution: bool, score: float, num_solns: int)
        self.solutions = [None] * self.num_refs
        # the vector needed to get the reference and moving images to overlap (or so we hope)
        self.adjustment_vectors = [None] * self.num_refs

        # reference data
        # ASSUME: all frames are identical in size. This is a rectangle that defines the size of a single full frame.
        self.full_frame = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))

        # extract the initial template data
        no_overlap = True
        for i in range(self.num_refs):
            if self.guess_template(i):
                no_overlap = False

        self.no_overlap = no_overlap

        # other state
        self.best_mses = [1e100] * self.num_refs
        self.best_matches = [(0, 0)] * self.num_refs

    def index_range(self):
        return range(self.num_refs)
    def num_indices(self):
        return self.num_refs
    def match_pt(self, index):
        return self.match_pts[index]
    def update_match_pt(self, index, match_pt):
        self.match_pts[index] = match_pt
    # returns the best MSE match seen so far. Only valid if we've done any MSE matching.
    def best_mse_match_pt(self, index):
        return self.best_matches[index]
    def adjustment_vector(self, index):
        return self.adjustment_vectors[index]

    # Guess a template for a given reference image index
    def guess_template(self, index):
        # Determine the nominal offsets based upon the machine's programmed x/y coordinates
        # for the image, based on the nominal stepping programmed into the imaging run.
        # For a perfect mechanical system:
        # moving_img + stepping_vector_px "aligns with" ref_img
        self.stepping_vectors[index] = Point(
            ((self.ref_metas[index]['x'] * 1000)
                - (self.moving_meta['x'] * 1000)) * Schema.PIX_PER_UM,
            ((self.ref_metas[index]['y'] * 1000)
                - (self.moving_meta['y'] * 1000)) * Schema.PIX_PER_UM
        )
        # a negative -x stepping vector means that the sample image is to the right of the reference image
        # a negative -y stepping vector means that the sample image is below the reference image

        # create an initial "template" based on the region of overlap between the reference and moving images
        self.intersection_rects[index] = self.full_frame.intersection(self.full_frame.translate(self.stepping_vectors[index]))
        if self.intersection_rects[index] is None:
            logging.warning(f"No overlap found between\   {self.ref_tiles[index]},\n   {self.moving_tile}")
            return False # no overlap at all

        # scale down the intersection template so we have a search space:
        # It's a balance between search space (where you can slide the template around)
        # and specificity (bigger template will have a chance of encapsulating more features)
        if False:
            self.template_rects[index] = self.intersection_rects[index].scale(SEARCH_SCALE)
        else:
            # turns out a smaller, square template works better in general?
            squared_region = self.intersection_rects[index].scale(SEARCH_SCALE).to_square()
            # heuristic: slide the template "up" just a little bit because we generally have
            # more overlap toward the edge of the frame
            up_offset = squared_region.tl.y / 2
            self.template_rects[index] = squared_region.translate(Point(0, -up_offset))

        self.templates[index] = self.moving_img[
            round(self.template_rects[index].tl.y) : round(self.template_rects[index].br.y),
            round(self.template_rects[index].tl.x) : round(self.template_rects[index].br.x)
        ].copy()

        # trace the template's extraction point back to the moving rectangle's origin
        scale_offset = self.template_rects[index].tl - self.intersection_rects[index].tl
        self.template_backtrack[index] = (-self.stepping_vectors[index]).clamp_zero() + scale_offset
        return True

    def adjust_template(self, index, template_shift):
        self.template_rects[index] = self.template_rects[index].saturating_translate(template_shift, self.full_frame)
        self.templates[index] = self.moving_img[
            round(self.template_rects[index].tl.y) : round(self.template_rects[index].br.y),
            round(self.template_rects[index].tl.x) : round(self.template_rects[index].br.x)
        ].copy()
        # trace the template's extraction point back to the moving rectangle's origin
        scale_offset = self.template_rects[index].tl - self.intersection_rects[index].tl
        self.template_backtrack[index] = (-self.stepping_vectors[index]).clamp_zero() + scale_offset

    def update_adjustment_vector(self, index):
        self.adjustment_vectors[index] = Point(
            self.match_pts[index][0] - self.template_backtrack[index][0],
            self.match_pts[index][1] - self.template_backtrack[index][1]
        )
    # returns the match_pt that you would use to create an alignment as if no adjustments were made.
    def unadjusted_machine_match_pt(self, index):
        # the first Point is needed to compensate for the fact that self.template_backtrack is subtracted
        # from match_pts to create the adjustment vector, as seen in the routine immediately above ^^^^
        # the second Point is the "naive" offset based on the relative machine offsets of the two tiles
        return Point(self.template_backtrack[index][0], self.template_backtrack[index][1]) + \
            Point(
                round(
                    (self.ref_tiles[index]['offset'][0] - self.moving_tile['offset'][0]) * Schema.PIX_PER_UM),
                round(
                    (self.ref_tiles[index]['offset'][1] - self.moving_tile['offset'][1]) * Schema.PIX_PER_UM)
            )
    def get_after_vector(self, index):
        return self.stepping_vectors[index] - self.adjustment_vectors[index]
        # return Point(int(
        #     - round(self.adjustment_vectors[index].x) +
        #     round((
        #         self.ref_metas[index]['x'] * 1000 + self.ref_tiles[index]['offset'][0]
        #         - (self.moving_meta['x'] * 1000 + self.proposed_offsets[0])
        #     ) * Schema.PIX_PER_UM)),
        #     int(- round(self.adjustment_vectors[index].y) +
        #     round((
        #         self.ref_metas[index]['y'] * 1000 + self.ref_tiles[index]['offset'][1]
        #         - (self.moving_meta['y'] * 1000 + self.proposed_offsets[1])
        #     ) * Schema.PIX_PER_UM))
        # )

    # Compute an auto-alignment against a given index
    def auto_align(self, index):
        if self.templates[index] is not None:
            # recompute all the images every call because the FILTER_WINDOW and LAPLACIAN_WINDOW can change on us
            if Schema.FILTER_WINDOW > 0:
                moving_norm = cv2.GaussianBlur(self.moving_img, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                moving_norm = moving_norm.astype(np.float32)
            else:
                moving_norm = cv2.normalize(self.moving_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            if Schema.FILTER_WINDOW > 0:
                ref_norm = cv2.GaussianBlur(self.ref_imgs[index], (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                template_norm = cv2.GaussianBlur(self.templates[index], (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                ref_norm = ref_norm.astype(np.float32)
                template_norm = template_norm.astype(np.float32)
            else: # normalize instead of filter
                ref_norm = cv2.normalize(self.ref_imgs[index], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                template_norm = cv2.normalize(self.templates[index], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # find edges
            self.ref_laplacians[index] = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
            template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

            # apply template matching. If the ref image and moving image are "perfectly aligned",
            # the value of `match_pt` should be equal to `template_ref`
            # i.e. alignment criteria: match_pt - template_ref = (0, 0)
            if True:
                METHOD = cv2.TM_CCOEFF  # convolutional matching
                res = cv2.matchTemplate(self.ref_laplacians[index], template_laplacian, METHOD)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                self.match_pts[index] = max_loc
                self.convolutions[index] = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                ret, thresh = cv2.threshold(self.convolutions[index], 224, 255, 0)
            else:
                METHOD = cv2.TM_SQDIFF  # squared error matching - not as good as convolutional matching for our purposes
                res = cv2.matchTemplate(self.ref_laplacians[index], template_laplacian, METHOD)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                match_pt = min_loc
                self.convolutions[index] = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.convolutions[index] = 255 - self.convolutions[index] # invert the thresholding
                ret, thresh = cv2.threshold(self.convolutions[index], 224, 255, 0)

            # find contours of candidate matches
            self.contours[index], self.hierarchies[index] = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            has_single_solution = True
            score = None
            num_solns = len(self.hierarchies[index][0])
            for i, c in enumerate(self.contours[index]):
                if self.hierarchies[index][0][i][3] == -1 and num_solns < MAX_SOLUTIONS:
                    if cv2.pointPolygonTest(c, self.match_pts[index], False) >= 0.0: # detect if point is inside or on the contour. On countour is necessary to detect cases of exact matches.
                        if score is not None:
                            has_single_solution = False
                        score = cv2.contourArea(c)
                        logging.debug(f"countour {c} contains {self.match_pts[index]} and has area {score}")
                        logging.debug(f"                    score: {score}")
                    else:
                        # print(f"countour {c} does not contain {top_left}")
                        pass
                else:
                    if cv2.pointPolygonTest(c, self.match_pts[index], False) > 0:
                        logging.debug(f"{self.match_pts[index]} of {i} is contained within a donut-shaped region. Suspect blurring error!")
                        has_single_solution = False
            if score is not None:
                logging.info(f"{i}: {score:0.1f} ({num_solns})")
            else:
                logging.info(f"{i}: No solution")
            self.solutions[index] = (has_single_solution, score, num_solns)
            self.update_adjustment_vector(index)

    def best_scoring_index(self):
        score = None
        picked = None
        for i, (ss, soln_score, num_solns) in enumerate(self.solutions):
            if ss is None:
                continue # skip this selection because it's not valid
            if score is None:
                picked = i
                score = soln_score
            else:
                if soln_score is not None:
                    if soln_score < score:
                        score = soln_score
                        picked = i
        return picked
    # this forces a valid score if one doesn't exist. Invoke during manual shifting
    # and "going with whatever the user picked"
    def force_score(self, index):
        self.solutions[index] = (True, FAILING_SCORE - 1, -1)

    # returns:
    #  'GOOD' if the proposed index meets our quality criteria for an auto-match
    #  'AMBIGUOUS' if the proposed index match quality is low
    #  'OUT_OF_RANGE' if the proposed index match is out of range
    def eval_index(self, index):
        (single_solution, score, num_solns) = self.solutions[index]
        if score is not None and single_solution and score < FAILING_SCORE:
            if abs(self.adjustment_vectors[index].x) > X_REVIEW_THRESH or abs(self.adjustment_vectors[index].y) > Y_REVIEW_THRESH:
                return f'OUT_OF_RANGE ({self.adjustment_vectors[index].x}, {self.adjustment_vectors[index].y})'
            else:
                return 'GOOD'
        else:
            return 'AMBIGUOUS'

    def has_single_solution(self, index):
        (single_solution, score, num_solns) = self.solutions[index]
        return single_solution
    def score(self, index):
        (single_solution, score, num_solns) = self.solutions[index]
        return score
    def num_solutions(self, index):
        (single_solution, score, num_solns) = self.solutions[index]
        return num_solns

    def show_contours(self, index):
        if self.convolutions[index] is not None and self.contours[index] is not None:
            cv2.drawContours(self.convolutions[index], self.contours[index], -1, (0,255,0), 1)
            cv2.imshow('contours', cv2.resize(self.convolutions[index], None, None, 0.5, 0.5))

    def score_as_str(self, index):
        msg = ''
        (single_solution, score, num_solns) = self.solutions[index]
        if score is not None:
            msg += f' score {score:0.1f}, {num_solns} solns'
        else:
            msg += f' score NONE'
        if single_solution != True:
            msg += ' (donut topology)'
        return msg

    def show_before_after(self, index, msg = ''):
        # compute after vector without storing the result
        after_vector_px = self.get_after_vector(index)
        ref_overlap = self.full_frame.intersection(
            self.full_frame.translate(Point(0, 0) - after_vector_px)
        )
        moving_overlap = self.full_frame.intersection(
            self.full_frame.translate(after_vector_px)
        )
        if ref_overlap is None or moving_overlap is None:
            after = np.zeros((500, 500), dtype=np.uint8)
        else:
            after = np.hstack(pad_images_to_same_size(
                (
                    # int() not round() used because python's "banker's rounding" sucks.
                    cv2.resize(self.ref_imgs[index][
                        int(ref_overlap.tl.y) : int(ref_overlap.br.y),
                        int(ref_overlap.tl.x) : int(ref_overlap.br.x)
                    ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE),
                    cv2.resize(self.moving_img[
                        int(moving_overlap.tl.y):int(moving_overlap.br.y),
                        int(moving_overlap.tl.x):int(moving_overlap.br.x)
                    ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE)
                )
            ))
        cv2.putText(
            after, msg,
            org=(25, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.putText(
            after, 'REF',
            org=(25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.putText(
            after, 'SAMPLE (aligned)',
            org=(after.shape[1] // 2 + 25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        if False:
            overview = np.hstack(pad_images_to_same_size(
                (
                    cv2.resize(ref_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                    cv2.resize(moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                )
            ))
        else:
            overview = cv2.normalize(
                np.hstack(pad_images_to_same_size(
                    (
                        cv2.resize(self.ref_imgs[index], None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                        cv2.resize(self.moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                    )
                )),
                None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        cv2.putText(
            overview, 'REF (full image, normalized)',
            org=(25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.putText(
            overview, 'SAMPLE (full image, unaligned, normalized)',
            org=(overview.shape[1] // 2 + 25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        before_after = np.vstack(
            pad_images_to_same_size(
                (cv2.resize(self.templates[index], None, None, PREVIEW_SCALE, PREVIEW_SCALE),
                after, overview)
            )
        )
        cv2.imshow('before/after', before_after)

    def show_stitch_preview(self, index, msg = ''):
        after_vector = self.get_after_vector(index)
        ref_canvas = np.zeros(
            (self.ref_imgs[index].shape[0] + ceil(abs(after_vector.y)),
             self.ref_imgs[index].shape[1] + ceil(abs(after_vector.x))
            ), dtype=np.uint8
        )
        moving_canvas = np.zeros(
            (self.ref_imgs[index].shape[0] + ceil(abs(after_vector.y)),
             self.ref_imgs[index].shape[1] + ceil(abs(after_vector.x))
            ), dtype=np.uint8
        )
        ref_orig = Point(0, 0)
        moving_orig = Point(0, 0)
        rh, rw = self.ref_imgs[index].shape
        if after_vector.x >= 0:
            ref_orig.x = round(after_vector.x)
            moving_orig.x = 0
        else:
            ref_orig.x = 0
            moving_orig.x = round(-after_vector.x)
        if after_vector.y >= 0:
            ref_orig.y = round(after_vector.y)
            moving_orig.y = 0
        else:
            ref_orig.y = 0
            moving_orig.y = round(-after_vector.y)
        ref_canvas[
            ref_orig.y : ref_orig.y + rh,
            ref_orig.x : ref_orig.x + rw
        ] = self.ref_imgs[index]
        moving_canvas[
            moving_orig.y : moving_orig.y + rh,
            moving_orig.x : moving_orig.x + rw
        ] = self.moving_img
        composite_canvas = cv2.addWeighted(ref_canvas, 0.5, moving_canvas, 0.5, 0.0)
        cv2.putText(
            composite_canvas, msg,
            org=(max([moving_orig.x, ref_orig.x]), max([moving_orig.y, ref_orig.y])),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2.0, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.imshow('stitch preview',
            cv2.resize(composite_canvas, None, None, 0.5, 0.5) # use different scale because resolution matters here
        )

    def reset_mse_tracker(self):
        self.best_mse = 1e100
        self.best_match = (0, 0)

    def show_mse(self, index):
        after_vector_px = self.get_after_vector(index)
        ref_overlap = self.full_frame.intersection(
            self.full_frame.translate(Point(0, 0) - after_vector_px)
        )
        moving_overlap = self.full_frame.intersection(
            self.full_frame.translate(after_vector_px)
        )
        # display the difference of laplacians of the overlapping region
        moving_roi = self.moving_img[
            round(moving_overlap.tl.y):round(moving_overlap.br.y),
            round(moving_overlap.tl.x):round(moving_overlap.br.x)
        ]
        ref_roi = self.ref_imgs[index][
            round(ref_overlap.tl.y) : round(ref_overlap.br.y),
            round(ref_overlap.tl.x) : round(ref_overlap.br.x)
        ]

        if Schema.FILTER_WINDOW > 0:
            ref_norm = cv2.GaussianBlur(ref_roi, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
            moving_norm = cv2.GaussianBlur(moving_roi, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
            ref_norm = ref_norm.astype(np.float32)
            moving_norm = moving_norm.astype(np.float32)
        else:
            ref_norm = cv2.normalize(ref_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            moving_norm = cv2.normalize(moving_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
        moving_laplacian = cv2.Laplacian(moving_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

        corr = moving_laplacian - ref_laplacian
        err = np.sum(corr**2)
        h, w = ref_laplacian.shape
        mse = err / (float(h*w))
        log_mse = round(log10(mse), 4)
        if log_mse <= self.best_mses[index]:
            self.best_mses[index] = log_mse
            mse_hint = 'best'
            self.best_matches[index] = self.match_pts[index]
        else:
            mse_hint = ''
        logging.info(f"MSE: {log_mse} {mse_hint}")
        corr_f32 = cv2.normalize(corr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        (_retval, corr_f32) = cv2.threshold(corr_f32, 0.8, 0.8, cv2.THRESH_TRUNC) # toss out extreme outliers (e.g. bad pixels)
        corr_f32 = cv2.normalize(corr_f32, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # corr_u8 = cv2.normalize(corr_f32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.putText(
            corr_f32, f"MSE: {log_mse} {mse_hint}",
            org=(100, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
        )
        cv2.imshow("Find minimum MSE", cv2.resize(corr_f32, None, None, 0.5, 0.5))

# Use template matching of laplacians to do stitching
def stitch_one_template(self,
                        schema,
                        ref_layers,
                        moving_layer,
                        retrench = False,
                        full_review = False,
    ):
    state = StitchState(schema, ref_layers, moving_layer)
    if state.no_overlap: # render an error message, and skip the stitching
        self.schema.store_auto_align_result(moving_layer, None, False, solutions=0)
        logging.warning("No overlap between any reference and moving frame")
        overview = cv2.resize(
            np.vstack(pad_images_to_same_size([state.moving_img].append(state.ref_imgs))),
            None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2
        )
        cv2.putText(
            overview, f"NO OVERLAP:\n{moving_layer}\n{ref_layers}",
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.imshow('before/after', overview)
        cv2.waitKey() # pause because no delay is specified
        return removed

    removed = False

    # compute all the initial stitching guesses
    for i in state.index_range():
        state.auto_align(i)

    #### Stitching interaction loop
    # options are 'AUTO', 'MOVE', 'TEMPLATE' or None
    #  - 'AUTO' is to try the full auto path
    #  - 'MOVE' is manual moving of the image itself
    #  - 'TEMPLATE' is adjusting the template and retrying the autostitch
    #  - None means to quit
    #
    # Note that the initial mode *must* be 'AUTO' because the first pass
    # sets up a number of intermediates relied upon by the adjustment routines.

    mode = 'AUTO'
    picked = state.best_scoring_index()
    while mode is not None:
        logging.info(f"picked: {picked}")
        msg = ''
        if mode == 'TEMPLATE':
            state.auto_align(picked)

        ##### Compute the putative adjustment vector
        state.update_adjustment_vector(picked)

        #### Decide if user feedback is needed
        # use the contours and the matched point to measure the quality of the template match
        if retrench:
            logging.info("Manual QC flagged")
            msg = 'Manual QC flagged'
            mode = 'TEMPLATE'
            retrench = False # so we don't keep triggering this in the loop
        elif mode == 'AUTO':
            verdict = state.eval_index(picked)
            if verdict == 'GOOD':
                if full_review:
                    msg = 'Good'
                    mode = 'TEMPLATE' # force a review of everything
                else:
                    msg = ''
                    mode = None
                    cv2.waitKey(10) # update UI, and move on without a pause
            elif verdict == 'AMBIGUOUS':
                msg = 'Ambiguous'
                mode = 'TEMPLATE'
            elif verdict == 'OUT_OF_RANGE':
                msg = 'Out of range'
                mode = 'TEMPLATE'
            else:
                logging.error(f"Internal error: unrecognized verdict on match: '{verdict}'")
                msg = 'Internal Error!'
                mode = 'TEMPLATE'

        ##### Render user feedback
        state.show_contours(picked)
        state.show_before_after(picked)
        msg += state.score_as_str(picked)
        state.show_stitch_preview(picked, msg)

        ##### Handle UI cases
        if mode == 'MOVE':
            state.show_mse(picked)
            # get user feedback and adjust the match_pt accordingly
            logging.warning(f"MOVE IMAGE: 'wasd' to move, space to accept, 1 to toggle to template mode, x to remove from database, y to snap to best point, Y to zero match pt")
            key = cv2.waitKey()
            COARSE_MOVE = 20
            if key != -1:
                key_char = chr(key)
                logging.debug(f'Got key: {key_char}')
                if key_char == ',' or key_char == 'w':
                    new_match_pt = (state.match_pt(picked)[0] + 0, state.match_pt(picked)[1] - 1)
                elif key_char == 'a':
                    new_match_pt = (state.match_pt(picked)[0] - 1, state.match_pt(picked)[1] + 0)
                elif key_char == 'e' or key_char == 'd':
                    new_match_pt = (state.match_pt(picked)[0] + 1, state.match_pt(picked)[1] + 0)
                elif key_char == 'o' or key_char == 's':
                    new_match_pt = (state.match_pt(picked)[0] + 0, state.match_pt(picked)[1] + 1)
                # coarse movement
                elif key_char == '<' or key_char == 'W':
                    new_match_pt = (state.match_pt(picked)[0] + 0, state.match_pt(picked)[1] - COARSE_MOVE)
                elif key_char == 'A':
                    new_match_pt = (state.match_pt(picked)[0] - COARSE_MOVE, state.match_pt(picked)[1] + 0)
                elif key_char == 'E' or key_char == 'D':
                    new_match_pt = (state.match_pt(picked)[0] + COARSE_MOVE, state.match_pt(picked)[1] + 0)
                elif key_char == 'O' or key_char == 'S':
                    new_match_pt = (state.match_pt(picked)[0] + 0, state.match_pt(picked)[1] + COARSE_MOVE)
                # reset match point to what a perfect machine would yield
                elif key_char == 'Y' or key_char == 'T':
                    new_match_pt = state.unadjusted_machine_match_pt(picked)
                    state.reset_mse_tracker() # reset the MSE score since we're in a totally different regime
                elif key_char == 'y' or key_char == 't':
                    new_match_pt = state.best_mse_match_pt(picked)
                elif key_char == ' ': # this accepts the current alignment
                    mode = None
                    state.force_score(picked)
                elif key_char == 'x': # this rejects the alignment
                    self.schema.remove_tile(moving_layer)
                    logging.info(f"Removing tile {moving_layer} from the database")
                    removed = True
                    mode = None
                elif key_char == '1':
                    mode = 'TEMPLATE'
                elif key_char == '\t': # toggle to another solution
                    while True:
                        picked = (picked + 1) % state.num_indices()
                        if state.templates[picked] is not None: # make sure the picked index is not invalid
                            break
                    logging.info(f"Toggling to index {picked}")
                    continue # this is very important to ensure that the match_pt isn't side-effected incorrectly on the new solution
                else:
                    logging.debug(f"Unhandled key: {key_char}, ignoring")
                    continue

                state.update_match_pt(picked, new_match_pt)
        elif mode == 'TEMPLATE':
            hint = state.eval_index(picked)
            logging.info(f"{state.score_as_str(picked)} ({hint})")
            logging.info("press 'wasd' to adjust template region, space to accept, 1 to toggle to manual move, x to remove from db")
            key = cv2.waitKey() # pause because no delay is specified
            SHIFT_AMOUNT = 50
            if key != -1:
                key_char = chr(key)
                template_shift = None
                if key_char == ',' or key_char == 'w':
                    template_shift = Point(0, -SHIFT_AMOUNT)
                elif key_char == 'a':
                    template_shift = Point(-SHIFT_AMOUNT, 0)
                elif key_char == 'e' or key_char == 'd':
                    template_shift = Point(SHIFT_AMOUNT, 0)
                elif key_char == 'o' or key_char == 's':
                    template_shift = Point(0, SHIFT_AMOUNT)
                elif key_char == 'x':
                    self.schema.remove_tile(moving_layer)
                    logging.info(f"Removing tile {moving_layer} from the database")
                    mode = None
                    template_shift = None
                    removed = True
                elif key_char == ' ':
                    logging.info("Accepting alignment")
                    mode = None
                    template_shift = None
                elif key_char == '1':
                    mode = 'MOVE'
                    template_shift = None
                elif key_char == '\t': # toggle to another solution
                    while True:
                        picked = (picked + 1) % state.num_indices()
                        if state.templates[picked] is not None: # make sure the picked index is not invalid
                            break
                    logging.info(f"Toggling to index {picked}")
                    continue # this is very important to ensure that the match_pt isn't side-effected incorrectly on the new solution
                else:
                    continue

                if template_shift is not None:
                    state.adjust_template(picked, template_shift)

        # store the result if the mode is set to None, and the schema still contains the moving layer.
        if mode == None and not removed:
            # Exit loop: store the stitch result
            logging.debug(f"minima at: {new_match_pt}")
            logging.debug(f"before adjustment: {state.moving_tile['offset'][0]},{state.moving_tile['offset'][1]}")
            # now update the offsets to reflect this
            self.schema.adjust_offset(
                moving_layer,
                state.adjustment_vector(picked).x / Schema.PIX_PER_UM,
                state.adjustment_vector(picked).y / Schema.PIX_PER_UM
            )
            self.schema.store_auto_align_result(
                moving_layer,
                state.score(picked),
                not state.has_single_solution(picked),
                solutions=state.num_solutions(picked)
            )
            check_t = self.schema.schema['tiles'][str(moving_layer)]
            logging.info(f"after adjustment: {check_t['offset'][0]:0.2f}, {check_t['offset'][1]:0.2f} score: {state.score(picked)} candidates: {state.num_solutions(picked)}")
    return removed

# Returns True if the database was modified and we have to restart the stitching pass
def stitch_auto_template_linear(self):
    removed = False
    coords = self.schema.coords
    anchor = None
    for coord in coords:
        (layer, t) = self.schema.get_tile_by_coordinate(coord)
        if t is not None: # handle db deletions
            meta_t = Schema.meta_from_tile(t)
            if t['auto_error'] == 'anchor':
                anchor = Point(meta_t['x'], meta_t['y'])

    if anchor is None:
        logging.error("Set anchor before stitching")
        return removed

    # roll the coordinate lists until we are starting at the anchor layer
    for x_roll in range(len(self.schema.x_list)):
        x_list = np.roll(self.schema.x_list, -x_roll)
        if x_list[0] == anchor.x:
            if x_roll != 0:
                x_list = np.concatenate([x_list[:-x_roll], x_list[-x_roll:][::-1]])
            break
    for y_roll in range(len(self.schema.y_list)):
        y_list = np.roll(self.schema.y_list, -y_roll)
        if y_list[0] == anchor.y:
            if y_roll != 0:
                y_list = np.concatenate([y_list[:-y_roll], y_list[-y_roll:][::-1]])
            break

    last_y_top = anchor
    next_tile = None
    prev_x = None
    for x in x_list:
        # stitch the top of an x-column to the previous y-column
        cur_tile = last_y_top
        top_of_y = True
        # now stitch an x-column
        for y in y_list:
            next_tile = Point(x, y)
            if top_of_y: # stash the top of y for the next column to align against
                last_y_top = next_tile
                top_of_y = False
            if next_tile == cur_tile:
                # because the first tile we'd ever encounter should be the anchor!
                continue
            else:
                (ref_layer, ref_t) = self.schema.get_tile_by_coordinate(cur_tile)
                if ref_t is None: # handle db deletions
                    cur_tile = next_tile
                    continue
                (moving_layer, moving_t) = self.schema.get_tile_by_coordinate(next_tile)
                if moving_t is None:
                    continue
                if moving_t['auto_error'] == 'invalid' or moving_t['auto_error'] == 'true':
                    if moving_t['auto_error'] == 'true':
                        retrench=True
                    else:
                        retrench=False
                    logging.info(f"Stitching {cur_tile} to {next_tile}")
                    if prev_x is not None: # add the left tile as an option for stitching (stitching goes left-to-right)
                        (left_layer, left_t) = self.schema.get_tile_by_coordinate(Point(prev_x, y))
                        if left_t is not None:
                            ref_layers = [ref_layer, left_layer]
                        else:
                            ref_layers = [ref_layer]
                    else:
                        ref_layers = [ref_layer]
                    removed = self.stitch_one_template(
                        self.schema,
                        ref_layers,
                        moving_layer,
                        retrench=retrench,
                        full_review=True
                    )
                    if True: #retrench:
                        self.schema.overwrite() # possibly dubious call to save the schema every time after manual patch-up
                    if removed:
                        return removed
            cur_tile = next_tile
        prev_x = x

    logging.info("Auto-stitch pass done")
    return removed

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
