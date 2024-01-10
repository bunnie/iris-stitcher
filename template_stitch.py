from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging
from itertools import combinations
from utils import pad_images_to_same_size
from math import log10, ceil, floor
from datetime import datetime
from progressbar.bar import ProgressBar
import threading

# low scores are better. scores greater than this fail.
FAILING_SCORE = 80.0
CONTOUR_THRESH = 192 # 192 for well-focused images; 224 if the focus quality is poor
# maximum number of potential solutions before falling back to manual review
MAX_SOLUTIONS = 8
PREVIEW_SCALE = 0.3
X_REVIEW_THRESH_UM = 80.0
Y_REVIEW_THRESH_UM = 80.0
SEARCH_SCALE = 0.80  # 0.8 worked on the AW set, 0.9 if using a square template
MAX_TEMPLATE_PX = 768

# Usage:
# Create the Profiler() object at the point where you want benchmarking to start
#
# Call profiler.lap('checkpoint name') at every point you want benchmarked. For
# best results, add a dummy 'top' and 'end' checkpoint at the top and end of the
# loop or routine.
#
# Call profiler.stats() to print the results.
class Profiler():
    def __init__(self):
        self.last_time = datetime.now()
        self.timers = {
        }

    def lap(self, id):
        current_time = datetime.now()
        if id in self.timers:
            delta = (current_time - self.last_time)
            self.timers[id] += delta
        else:
            self.timers[id] = current_time - self.last_time
        self.last_time = current_time

    def stats(self):
        sorted_times = dict(sorted(self.timers.items(), key=lambda x: x[1], reverse=True))
        for i, (id, delta) in enumerate(sorted_times.items()):
            logging.info(f'    {id}: {delta}')
            if i >= 2: # just list the top 3 items for now
                break

class StitchState():
    @staticmethod
    def alloc_nested(inner, outer):
        ret = []
        for _o in range(outer):
            ret += [
                [None for _m in range(inner)]
            ]
        return ret
    @staticmethod
    def alloc_list(length):
        return [None for _m in range(length)]

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
            self.ref_imgs += [self.schema.get_image_from_layer(ref_layer)]
            self.ref_tiles += [tile]

        tile = self.schema.schema['tiles'][moving_layer]
        assert tile is not None, f"The layer to be stitched {moving_layer} is missing!"
        self.moving_img = self.schema.get_image_from_layer(moving_layer)
        self.moving_meta = Schema.meta_from_tile(tile)
        self.moving_tile = tile

        ### derived data
        self.num_refs = len(ref_layers)
        # the proposed offset for the moving image for a given ref layer
        self.stepping_vectors = [None] * self.num_refs
        # best and current templates from the list of possible templates for a given index
        self.best_templates = [None] * self.num_refs
        # Current Template Selected list - shortened name because it's used everywhere
        self.cts = [None] * self.num_refs
        # the actual pixel data of the template used for template matching
        self.templates = [[None] for x in range(self.num_refs)]
        # the rectangle that defines the template, relative to the moving image origin
        self.template_rects = [[None] for x in range(self.num_refs)]
        # the backtrack vector for the template - the vector needed to reverse the match point into an image offset
        self.template_backtrack = [[None] for x in range(self.num_refs)] # was named 'template_refs'
        # the full region of intersection between the ref & moving images
        self.intersection_rects = [None] * self.num_refs
        self.ref_laplacians = [None] * self.num_refs
        # the contours themselves
        self.contours = [[None] for x in range(self.num_refs)]
        # hierarchy of contours that match (to discover nested solutions, etc.)
        self.hierarchies = [[None] for x in range(self.num_refs)]
        # best match based on the current template convolution
        self.match_pts = [[None] for x in range(self.num_refs)]
        # template convolved with reference image
        self.convolutions = [[None] for x in range(self.num_refs)] # was named 'results'
        # tuple of (has_single_solution: bool, score: float, num_solns: int)
        self.solutions = [[None] for x in range(self.num_refs)]
        # the vector needed to get the reference and moving images to overlap (or so we hope)
        self.adjustment_vectors = [[None] for x in range(self.num_refs)]

        # reference data
        # ASSUME: all frames are identical in size. This is a rectangle that defines the size of a single full frame.
        self.full_frame = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))

        # extract the initial template data
        no_overlap = True
        for i in range(self.num_refs):
            if self.guess_template(i, multiple=True):
                no_overlap = False

        self.no_overlap = no_overlap

        # other state
        self.best_mses = [1e100] * self.num_refs
        self.best_matches = [(0, 0)] * self.num_refs

    def index_range(self):
        return range(self.num_refs)
    def num_indices(self):
        return self.num_refs
    def match_pt(self, i):
        return self.match_pts[i][self.cts[i]]
    def update_match_pt(self, i, match_pt):
        self.match_pts[i][self.cts[i]] = match_pt
    # returns the best MSE match seen so far. Only valid if we've done any MSE matching.
    def best_mse_match_pt(self, i):
        return self.best_matches[i]
    def adjustment_vector(self, index):
        return self.adjustment_vectors[index][self.cts[index]]

    # Guess a template for a given reference image index
    def guess_template(self, i, multiple=False):
        # Determine the nominal offsets based upon the machine's programmed x/y coordinates
        # for the image, based on the nominal stepping programmed into the imaging run.
        # For a perfect mechanical system:
        # moving_img + stepping_vector_px "aligns with" ref_img
        self.stepping_vectors[i] = Point(
            ((self.ref_metas[i]['x'] * 1000)
                - (self.moving_meta['x'] * 1000)) * Schema.PIX_PER_UM,
            ((self.ref_metas[i]['y'] * 1000)
                - (self.moving_meta['y'] * 1000)) * Schema.PIX_PER_UM
        )
        # a negative -x stepping vector means that the sample image is to the right of the reference image
        # a negative -y stepping vector means that the sample image is below the reference image

        # create an initial "template" based on the region of overlap between the reference and moving images
        self.intersection_rects[i] = self.full_frame.intersection(self.full_frame.translate(self.stepping_vectors[i]))
        if self.intersection_rects[i] is None:
            logging.warning(f"No overlap found between\   {self.ref_tiles[i]},\n   {self.moving_tile}")
            return False # no overlap at all

        # set the current selection and best to the 0th index, as a starting point
        self.best_templates[i] = 0
        self.cts[i] = 0

        if multiple is False:
            # scale down the intersection template so we have a search space:
            # It's a balance between search space (where you can slide the template around)
            # and specificity (bigger template will have a chance of encapsulating more features)
            if False:
                self.template_rects[i][self.cts[i]] = self.intersection_rects[i].scale(SEARCH_SCALE)
            else:
                # turns out a smaller, square template works better in general?
                squared_region = self.intersection_rects[i].scale(SEARCH_SCALE).to_square()
                # heuristic: slide the template "up" just a little bit because we generally have
                # more overlap toward the edge of the frame
                if False:
                    up_offset = squared_region.tl.y / 2
                    self.template_rects[i][self.cts[i]] = squared_region.translate(Point(0, -up_offset))
                else:
                    self.template_rects[i][self.cts[i]] = squared_region

            self.templates[i][self.cts[i]] = self.moving_img[
                round(self.template_rects[i][self.cts[i]].tl.y) \
                    : round(self.template_rects[i][self.cts[i]].br.y),
                round(self.template_rects[i][self.cts[i]].tl.x) \
                    : round(self.template_rects[i][self.cts[i]].br.x)
            ].copy()

            # trace the template's extraction point back to the moving rectangle's origin
            scale_offset = self.template_rects[i][self.cts[i]].tl - self.intersection_rects[i].tl
            self.template_backtrack[i][self.cts[i]] \
                = (-self.stepping_vectors[i]).clamp_zero() + scale_offset
        else:
            intersection_w = self.intersection_rects[i].width()
            intersection_h = self.intersection_rects[i].height()
            template_dim = intersection_w
            # find the smallest square that fits
            if template_dim > intersection_h:
                template_dim = intersection_h
            template_dim = template_dim * SEARCH_SCALE
            # limit the max template dimension
            if template_dim > MAX_TEMPLATE_PX:
                template_dim = MAX_TEMPLATE_PX
            # adjust for high aspect ratio situations, but preferring square
            if intersection_w / intersection_h > 2:
                template_dim_x = template_dim * 2
                template_dim_y = template_dim
            elif intersection_w / intersection_h < 2:
                template_dim_x = template_dim
                template_dim_y = template_dim * 2
            else:
                template_dim_x = template_dim
                template_dim_y = template_dim

            # allocate placeholders for all the templates
            x_range = range(int(self.intersection_rects[i].tl.x), int(self.intersection_rects[i].br.x), int(template_dim_x // 2))
            y_range = range(int(self.intersection_rects[i].tl.y), int(self.intersection_rects[i].br.y), int(template_dim_y // 2))
            self.templates[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.template_rects[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.template_backtrack[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.contours[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.hierarchies[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.match_pts[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.convolutions[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.solutions[i] = StitchState.alloc_list(len(x_range) * len(y_range))
            self.adjustment_vectors[i] = StitchState.alloc_list(len(x_range) * len(y_range))

            templ_index = 0
            for x in x_range:
                if x + template_dim_x >= self.intersection_rects[i].br.x:
                    x = self.intersection_rects[i].br.x - template_dim_x # align last iter to right edge
                for y in y_range:
                    if y + template_dim_y >= self.intersection_rects[i].br.y:
                        y = self.intersection_rects[i].br.y - template_dim_y # align last iter to bottom edge

                    template_rect = Rect(
                        Point(int(x), int(y)),
                        Point(int(x + template_dim_x), int(y + template_dim_y))
                    )
                    self.template_rects[i][templ_index] = template_rect
                    self.templates[i][templ_index] = self.moving_img[
                        template_rect.tl.y : template_rect.br.y,
                        template_rect.tl.x : template_rect.br.x
                    ].copy()
                    scale_offset = template_rect.tl - self.intersection_rects[i].tl
                    self.template_backtrack[i][templ_index] = (-self.stepping_vectors[i]).clamp_zero() + scale_offset
                    templ_index += 1

        return True

    def adjust_template(self, i, template_shift):
        self.template_rects[i][self.cts[i]] = self.template_rects[i][self.cts[i]].saturating_translate(template_shift, self.full_frame)
        self.templates[i][self.cts[i]] = self.moving_img[
            round(self.template_rects[i][self.cts[i]].tl.y) \
                : round(self.template_rects[i][self.cts[i]].br.y),
            round(self.template_rects[i][self.cts[i]].tl.x) \
                : round(self.template_rects[i][self.cts[i]].br.x)
        ].copy()
        # trace the template's extraction point back to the moving rectangle's origin
        scale_offset = self.template_rects[i][self.cts[i]].tl - self.intersection_rects[i].tl
        self.template_backtrack[i][self.cts[i]] \
            = (-self.stepping_vectors[i]).clamp_zero() + scale_offset

    def update_adjustment_vector(self, i, template_index = None):
        if template_index is None:
            t = self.cts[i]
        else:
            t = template_index
        self.adjustment_vectors[i][t] = Point(
            self.match_pts[i][t][0] - self.template_backtrack[i][t][0],
            self.match_pts[i][t][1] - self.template_backtrack[i][t][1]
        )

    # returns the match_pt that you would use to create an alignment as if no adjustments were made.
    def unadjusted_machine_match_pt(self, i):
        # the first Point is needed to compensate for the fact that self.template_backtrack is subtracted
        # from match_pts to create the adjustment vector, as seen in the routine immediately above ^^^^
        # the second Point is the "naive" offset based on the relative machine offsets of the two tiles
        return Point(self.template_backtrack[i][self.cts[i]][0],
                     self.template_backtrack[i][self.cts[i]][1]) \
            + Point(
                round(
                    (self.ref_tiles[i]['offset'][0] - self.moving_tile['offset'][0]) * Schema.PIX_PER_UM),
                round(
                    (self.ref_tiles[i]['offset'][1] - self.moving_tile['offset'][1]) * Schema.PIX_PER_UM)
            )

    def get_after_vector(self, i):
        return self.stepping_vectors[i] - self.adjustment_vectors[i][self.cts[i]]

    # computes the final offset to store in the database. Also has to sum in the offset
    # of the reference image, because we did all the computations based on 0-offset on
    # the reference image. Returns a coordinate in Pixels, has to be converted to microns
    # for storage into the Schema.
    def finalize_offset(self, i):
        self.update_adjustment_vector(i) # this should already be done...? should we do it again?
        return Point(
            self.adjustment_vectors[i][self.cts[i]].x + self.ref_tiles[i]['offset'][0] * Schema.PIX_PER_UM,
            self.adjustment_vectors[i][self.cts[i]].y + self.ref_tiles[i]['offset'][1] * Schema.PIX_PER_UM,
        )

    # The auto-alignment kernel. This code must be thread-safe. The thread safety is "ensured"
    # by having every operation only refer to its assigned (i,t) slot in memory. So long as
    # (i, t) are unique, we should have no collisions.
    def auto_align_kernel(self, template, i, t):
        if Schema.FILTER_WINDOW > 0:
            template_norm = cv2.GaussianBlur(template, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
            template_norm = template_norm.astype(np.float32)
        else: # normalize instead of filter
            template_norm = cv2.normalize(template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # find edges
        template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

        # apply template matching. If the ref image and moving image are "perfectly aligned",
        # the value of `match_pt` should be equal to `template_ref`
        # i.e. alignment criteria: match_pt - template_ref = (0, 0)
        METHOD = cv2.TM_CCOEFF  # convolutional matching
        res = cv2.matchTemplate(self.ref_laplacians[i], template_laplacian, METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        self.match_pts[i][t] = max_loc
        self.convolutions[i][t] = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ret, thresh = cv2.threshold(self.convolutions[i][t], CONTOUR_THRESH, 255, 0)

        # find contours of candidate matches
        self.contours[i][t], self.hierarchies[i][t] = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        has_single_solution = True
        score = None
        num_solns = len(self.hierarchies[i][t][0])
        if False:
            for j, c in enumerate(self.contours[i][t]):
                if self.hierarchies[i][t][0][j][3] == -1 and num_solns < MAX_SOLUTIONS:
                    if cv2.pointPolygonTest(c, self.match_pts[i][t], False) >= 0.0: # detect if point is inside or on the contour. On countour is necessary to detect cases of exact matches.
                        if score is not None:
                            has_single_solution = False
                        score = cv2.contourArea(c)
                        logging.debug(f"countour {c} contains {self.match_pts[i][t]} and has area {score}")
                        logging.debug(f"                    score: {score}")
                    else:
                        # print(f"countour {c} does not contain {top_left}")
                        pass
                else:
                    if cv2.pointPolygonTest(c, self.match_pts[i][t], False) > 0:
                        logging.debug(f"{self.match_pts[i][t]} of {i} is contained within a donut-shaped region. Suspect blurring error!")
                        has_single_solution = False
        else:
            for j, c in enumerate(self.contours[i][t]):
                if cv2.pointPolygonTest(c, self.match_pts[i][t], False) >= 0.0: # detect if point is inside or on the contour. On countour is necessary to detect cases of exact matches.
                    score = cv2.contourArea(c)
                    logging.debug(f"countour {c} contains {self.match_pts[i][t]} and has area {score}")
                    logging.debug(f"                    score: {score}")
                else:
                    # print(f"countour {c} does not contain {top_left}")
                    pass
        self.solutions[i][t] = (has_single_solution, score, num_solns)
        if score is not None:
            self.update_adjustment_vector(i, template_index=t)

    # Compute an auto-alignment against a given index
    def auto_align(self, i, multiple=False, multithreading=False):
        if multiple is True:
            template_range = range(len(self.templates[i]))
            if not multithreading:
                progress = ProgressBar(min_value=0, max_value=len(template_range), prefix='Matching ').start()
        else:
            template_range = range(self.cts[i], self.cts[i] + 1) # just 'iterate' over that one index

        # recompute every call because the FILTER_WINDOW and LAPLACIAN_WINDOW can change on us
        if Schema.FILTER_WINDOW > 0:
            moving_norm = cv2.GaussianBlur(self.moving_img, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
            moving_norm = moving_norm.astype(np.float32)
        else:
            moving_norm = cv2.normalize(self.moving_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if Schema.FILTER_WINDOW > 0:
            ref_norm = cv2.GaussianBlur(self.ref_imgs[i], (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
            ref_norm = ref_norm.astype(np.float32)
        else:
            ref_norm = cv2.normalize(self.ref_imgs[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.ref_laplacians[i] = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

        if multithreading:
            threads = []
            for t in template_range:
                if self.templates[i][t] is not None:
                    thread = threading.Thread(target=self.auto_align_kernel, args=(self.templates[i][t], i, t))
                    thread.start()
                    threads += [thread]

            for thread in threads:
                thread.join()
        else:
            for t in template_range:
                if self.templates[i][t] is not None:
                    self.auto_align_kernel(self.templates[i][t], i, t)
                    if multiple:
                        progress.update(t)
            if multiple:
                progress.finish()

    # computes the best score at the current template for each ref image
    def best_scoring_index(self, multiple=False):
        if multiple:
            # go through each ref index and search the entire space of templates for the best match.
            # best template is automatically selected by this, for each ref index
            score = None
            solns = None
            picked = None
            for i, solution_list in enumerate(self.solutions):
                if self.intersection_rects[i] is None:
                    continue # can't score things that didn't intersect
                best_template = None
                for template_index, (ss, soln_score, num_solns) in enumerate(solution_list):
                    if ss is None:
                        continue # skip this selection because it's not valid
                    if soln_score is not None and round(soln_score, 1) == 0.0:
                        # HEURISITIC: extremely low score of 0 usually means the match is a normalization
                        # artifact -- nothing was really matching, this just happens to be the best matching
                        # point in a region of "meh"-ness. Reject these points.
                        continue
                    if score is None:
                        picked = i
                        best_template = template_index
                        score = soln_score
                        solns = num_solns
                    else:
                        if soln_score is not None:
                            if score < FAILING_SCORE:
                                if num_solns > solns:
                                    # HEURISTIC: reject any solution that is less unique, so long as
                                    # our base score isn't failing
                                    continue
                            if soln_score < score:
                                score = soln_score
                                picked = i
                                best_template = template_index
                                solns = num_solns
                if best_template is not None:
                    self.best_templates[i] = best_template
                    self.cts[i] = best_template
                # else stick with the default, e.g. 0
            return picked
        else:
            score = None
            picked = None
            for i, solution_list in enumerate(self.solutions):
                (ss, soln_score, num_solns) = solution_list[self.cts[i]]
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
    def force_score(self, i):
        self.solutions[i][self.cts[i]] = (True, FAILING_SCORE - 1, -1)

    # returns:
    #  'GOOD' if the proposed index meets our quality criteria for an auto-match
    #  'AMBIGUOUS' if the proposed index match quality is low
    #  'OUT_OF_RANGE' if the proposed index match is out of range
    def eval_index(self, i):
        (single_solution, score, num_solns) = self.solutions[i][self.cts[i]]
        if score is not None and single_solution and score < FAILING_SCORE:
            if abs(self.adjustment_vectors[i][self.cts[i]].x) > X_REVIEW_THRESH_UM * Schema.PIX_PER_UM \
                or abs(self.adjustment_vectors[i][self.cts[i]].y) > Y_REVIEW_THRESH_UM * Schema.PIX_PER_UM:
                return f'OUT_OF_RANGE ({self.adjustment_vectors[i][self.cts[i]].x / Schema.PIX_PER_UM:0.1f}um, {self.adjustment_vectors[i][self.cts[i]].y / Schema.PIX_PER_UM:0.1f}um)'
            else:
                return 'GOOD'
        else:
            return 'AMBIGUOUS'

    def has_single_solution(self, i):
        (single_solution, score, num_solns) = self.solutions[i][self.cts[i]]
        return single_solution
    def score(self, i):
        (single_solution, score, num_solns) = self.solutions[i][self.cts[i]]
        return score
    def num_solutions(self, i):
        (single_solution, score, num_solns) = self.solutions[i][self.cts[i]]
        return num_solns

    def show_contours(self, i):
        if self.convolutions[i][self.cts[i]] is not None and self.contours[i][self.cts[i]] is not None:
            cv2.drawContours(self.convolutions[i][self.cts[i]], self.contours[i][self.cts[i]], -1, (0,255,0), 1)
            cv2.imshow('contours', cv2.resize(self.convolutions[i][self.cts[i]], None, None, 0.5, 0.5))

    def score_as_str(self, i):
        msg = ''
        (single_solution, score, num_solns) = self.solutions[i][self.cts[i]]
        if score is not None:
            msg += f' score {score:0.1f}, {num_solns} solns'
        else:
            msg += f' score NONE'
        if single_solution != True:
            msg += ' (donut topology)'
        return msg

    def show_debug(self, i, msg = ''):
        # compute after vector without storing the result
        after_vector_px = self.get_after_vector(i)
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
                    cv2.resize(self.ref_imgs[i][
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
                        cv2.resize(self.ref_imgs[i], None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
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
                (cv2.resize(self.templates[i][self.cts[i]], None, None, PREVIEW_SCALE, PREVIEW_SCALE),
                after, overview)
            )
        )
        cv2.imshow('debug', before_after)

    def show_stitch_preview(self, i, msg = ''):
        after_vector = self.get_after_vector(i)
        ref_canvas = np.zeros(
            (self.ref_imgs[i].shape[0] + ceil(abs(after_vector.y)),
             self.ref_imgs[i].shape[1] + ceil(abs(after_vector.x))
            ), dtype=np.uint8
        )
        moving_canvas = np.zeros(
            (self.ref_imgs[i].shape[0] + ceil(abs(after_vector.y)),
             self.ref_imgs[i].shape[1] + ceil(abs(after_vector.x))
            ), dtype=np.uint8
        )
        ref_orig = Point(0, 0)
        moving_orig = Point(0, 0)
        rh, rw = self.ref_imgs[i].shape
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
        ] = self.ref_imgs[i]
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

    def reset_mse_tracker(self, i):
        self.best_mses[i] = 1e100
        self.best_matches[i] = (0, 0)

    def show_mse(self, i):
        after_vector_px = self.get_after_vector(i)
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
        ref_roi = self.ref_imgs[i][
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
        if log_mse <= self.best_mses[i]:
            self.best_mses[i] = log_mse
            mse_hint = 'best'
            self.best_matches[i] = self.match_pts[i][self.cts[i]]
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
        return removed, False

    removed = False

    # compute all the initial stitching guesses
    for i in state.index_range():
        state.auto_align(i, multiple=True, multithreading=True)

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
    picked = state.best_scoring_index(multiple=True)
    while mode is not None:
        msg = ''
        if mode == 'TEMPLATE':
            state.auto_align(picked, multiple=False)

        ##### Compute the putative adjustment vector
        state.update_adjustment_vector(picked)

        #### Decide if user feedback is needed
        # use the contours and the matched point to measure the quality of the template match
        if retrench:
            logging.debug("Manual QC flagged")
            msg = 'Manual QC flagged'
            mode = 'TEMPLATE'
            retrench = False # so we don't keep triggering this in the loop
        elif mode == 'AUTO':
            verdict = state.eval_index(picked)
            if verdict == 'GOOD':
                if full_review:
                    msg = 'Good (edge)'
                    mode = 'TEMPLATE' # force a review of everything
                else:
                    msg = ''
                    mode = None
            elif verdict == 'AMBIGUOUS':
                msg = 'Ambiguous'
                mode = 'TEMPLATE'
            elif verdict.startswith('OUT_OF_RANGE'):
                msg = verdict
                mode = 'TEMPLATE'
            else:
                logging.error(f"Internal error: unrecognized verdict on match: '{verdict}'")
                msg = 'Internal Error!'
                mode = 'TEMPLATE'

        ##### Render user feedback
        state.show_contours(picked)
        state.show_debug(picked)
        msg += state.score_as_str(picked)
        state.show_stitch_preview(picked, msg)

        ##### Handle UI cases
        hint = state.eval_index(picked)
        logging.info(f"{picked}[{state.cts[picked]}]: {state.score_as_str(picked)} ({hint})]")
        if mode == 'MOVE':
            state.show_mse(picked)
            # get user feedback and adjust the match_pt accordingly
            logging.warning(f"MOVE IMAGE: 'wasd' to move, space to accept, 1 to toggle to template mode, x to remove from database, y to snap to best point, Y to zero match pt, 0 to abort run")
            key = cv2.waitKey()
            COARSE_MOVE = 20
            if key != -1:
                new_match_pt = None
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
                    state.update_adjustment_vector(picked)
                    new_match_pt = state.unadjusted_machine_match_pt(picked)
                    state.reset_mse_tracker(picked) # reset the MSE score since we're in a totally different regime
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
                        if state.intersection_rects[picked] is not None: # make sure the picked index is not invalid
                            break
                    logging.info(f"Toggling to index {picked}")
                    continue # this is very important to ensure that the match_pt isn't side-effected incorrectly on the new solution
                elif key_char == '0':
                    logging.info("Stitch abort!")
                    return False, True
                else:
                    logging.debug(f"Unhandled key: {key_char}, ignoring")
                    continue

                if new_match_pt is not None:
                    state.update_match_pt(picked, new_match_pt)
        elif mode == 'TEMPLATE':
            logging.info("press 'wasd' to adjust template region, space to accept, 1 to toggle to manual move, x to remove from db, 0 to abort run")
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
                    mode = None
                    template_shift = None
                elif key_char == '1':
                    mode = 'MOVE'
                    template_shift = None
                elif key_char == '\t': # toggle to another solution
                    while True:
                        picked = (picked + 1) % state.num_indices()
                        if state.intersection_rects[picked] is not None: # make sure the picked index is not invalid
                            break
                    logging.info(f"Toggling to index {picked}")
                    continue # this is very important to ensure that the match_pt isn't side-effected incorrectly on the new solution
                elif key_char == '0':
                    logging.info("Stitch abort!")
                    return False, True
                else:
                    continue

                if template_shift is not None:
                    state.adjust_template(picked, template_shift)

        # store the result if the mode is set to None, and the schema still contains the moving layer.
        if mode == None and not removed:
            cv2.waitKey(10) # update UI, and move on without a pause

            # Exit loop: store the stitch result
            logging.debug(f"minima at: {state.match_pt(picked)}")
            logging.debug(f"before adjustment: {state.moving_tile['offset'][0]},{state.moving_tile['offset'][1]}")
            # now update the offsets to reflect this
            finalized_pt = state.finalize_offset(picked)
            self.schema.adjust_offset(
                moving_layer,
                finalized_pt.x / Schema.PIX_PER_UM,
                finalized_pt.y / Schema.PIX_PER_UM
            )
            self.schema.store_auto_align_result(
                moving_layer,
                state.score(picked),
                not state.has_single_solution(picked),
                solutions=state.num_solutions(picked)
            )
            check_t = self.schema.schema['tiles'][str(moving_layer)]
            logging.info(f"after adjustment: {check_t['offset'][0]:0.2f}, {check_t['offset'][1]:0.2f} score: {state.score(picked)} candidates: {state.num_solutions(picked)}")
    return removed, False

# Returns True if the database was modified and we have to restart the stitching pass
def stitch_auto_template_linear(self, stitch_list=None):
    removed = False
    if stitch_list is None:
        coords = self.schema.coords
    else:
        coords = stitch_list
    anchor = None
    min_x = 1e10
    min_y = 1e10

    x_list = []
    y_list = []
    for coord in coords:
        (layer, t) = self.schema.get_tile_by_coordinate(coord)
        if t is not None: # handle db deletions
            meta_t = Schema.meta_from_tile(t)
            if meta_t['x'] < min_x:
                min_x = meta_t['x']
            if meta_t['y'] < min_y:
                min_y = meta_t['y']
            # anchor is the tile in the top left
            if meta_t['x'] == min_x and meta_t['y'] == min_y:
                anchor = Point(meta_t['x'], meta_t['y'])
            if meta_t['x'] not in x_list:
                x_list += [meta_t['x']]
            if meta_t['y'] not in y_list:
                y_list += [meta_t['y']]

    if anchor is None:
        logging.error("Couldn't find anchor!")
        return removed
    else:
        logging.info(f"Using anchor of {anchor}")

    x_list = sorted(x_list)
    y_list = sorted(y_list)

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
                edge_case = False
                (ref_layer, ref_t) = self.schema.get_tile_by_coordinate(cur_tile)
                if ref_t is None: # handle db deletions
                    cur_tile = next_tile
                    continue
                (moving_layer, moving_t) = self.schema.get_tile_by_coordinate(next_tile)
                if moving_t is None:
                    continue
                if moving_t['auto_error'] == 'invalid' or moving_t['auto_error'] == 'true' or stitch_list is not None:
                    if moving_t['auto_error'] == 'true':
                        retrench=True
                    else:
                        retrench=False
                    logging.info(f"Stitching {cur_tile} to {next_tile}")
                    if prev_x is not None: # add the left tile as an option for stitching (stitching goes left-to-right)
                        if y == last_y_top.y:
                            # on the top row, we actually want the left item and left-lower item as options, assuming
                            # we're not already at the very left
                            edge_case = True
                            (left_lower_layer, left_t) = self.schema.get_tile_by_coordinate(Point(prev_x, y + Schema.NOM_STEP))
                            if left_t is not None:
                                ref_layers = [ref_layer, left_lower_layer]
                            else:
                                ref_layers = [ref_layer]
                        else:
                            # on vertical stitches, grab the layer to the left and below
                            (left_layer, left_t) = self.schema.get_tile_by_coordinate(Point(prev_x, y))
                            if left_t is not None:
                                ref_layers = [ref_layer, left_layer]
                            else:
                                edge_case = True
                                ref_layers = [ref_layer]
                    else:
                        ref_layers = [ref_layer]
                        edge_case = True
                    # Right now, we force a full review on top and left edges. This is because
                    # The correctness of these alignments have to be spot-on for everything else to work,
                    # and also because on the edges there is a strong chance that a template match
                    # picks something *not* on the chip to match on (some of the background matting
                    # or an edge artifact of the chip, which is poorly focused). Haven't figured out a
                    # good way to figure out how to avoid that -- maybe some specialized algorithm that
                    # knows we're in an edge case and looks for stuff only to the inside of the brightest
                    # line that would define the chip edge?
                    removed, abort = self.stitch_one_template(
                        self.schema,
                        ref_layers,
                        moving_layer,
                        retrench=retrench,
                        full_review=edge_case
                    )
                    if abort:
                        return False
                    if True: #retrench:
                        self.schema.overwrite() # possibly dubious call to save the schema every time after manual patch-up
                    if removed:
                        return removed
            cur_tile = next_tile
        prev_x = x

    logging.info("Auto-stitch pass done")
    return removed

def restitch_one(self, moving_layer):
    # search for nearby tiles that have large overlaps and make them reference layers
    (moving_meta, moving_tile) = self.schema.get_info_from_layer(moving_layer)
    moving_x_mm = moving_meta['x']
    moving_y_mm = moving_meta['y']
    sorted_x = sorted(self.schema.x_list)
    sorted_y = sorted(self.schema.y_list)

    x_i = sorted_x.index(moving_x_mm)
    if x_i > 0:
        next_lower_x_mm = sorted_x[x_i - 1]
    else:
        next_lower_x_mm = None
    y_i = sorted_y.index(moving_y_mm)
    if y_i > 0:
        next_lower_y_mm = sorted_y[y_i - 1]
    else:
        next_lower_y_mm = None

    ref_layers = []
    # katty corner back
    if next_lower_y_mm is not None and next_lower_x_mm is not None:
        (layer, t) = self.schema.get_tile_by_coordinate((next_lower_x_mm, next_lower_y_mm))
        if layer is not None:
            if t['auto_error'] != 'false':
                logging.warning(f"Skipping candidate layer {layer} because it has a bad stitch")
            else:
                ref_layers += [layer]
    # up
    if next_lower_y_mm is not None:
        (layer, _t) = self.schema.get_tile_by_coordinate((moving_x_mm, next_lower_y_mm))
        if layer is not None:
            if t['auto_error'] != 'false':
                logging.warning(f"Skipping candidate layer {layer} because it has a bad stitch")
            else:
                ref_layers += [layer]
    # left
    if next_lower_x_mm is not None:
        (layer, _t) = self.schema.get_tile_by_coordinate((next_lower_x_mm, moving_y_mm))
        if layer is not None:
            if t['auto_error'] != 'false':
                logging.warning(f"Skipping candidate layer {layer} because it has a bad stitch")
            else:
                ref_layers += [layer]

    self.stitch_one_template(
        self.schema,
        ref_layers,
        moving_layer,
        retrench=False,
        full_review=True
    )
    logging.info("Done with restitch one...")
    self.schema.overwrite()

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
