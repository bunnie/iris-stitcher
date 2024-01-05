from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging
from itertools import combinations
from utils import pad_images_to_same_size
from math import log10, ceil

# low scores are better. scores greater than this fail.
FAILING_SCORE = 100.0
# maximum number of potential solutions before falling back to manual review
MAX_SOLUTIONS = 12
PREVIEW_SCALE = 0.3
X_REVIEW_THRESH = 100.0
Y_REVIEW_THRESH = 100.0

# Use template matching of laplacians to do stitching
def stitch_one_template(self,
                        ref_data,
                        moving_img, moving_meta, moving_t,
                        moving_layer,
                        retrench = False,
                        full_review = False,
    ):
    removed = False
    # ASSUME: all frames are identical in size. This is a rectangle that defines the size of a single full frame.
    full_frame = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))

    templates = []
    template_rects = []
    template_rects_full = []
    template_refs = []
    stepping_vectors = []

    #### extract stitching templates
    no_overlaps = True
    for ref_img, ref_meta, ref_t in ref_data:
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
            stepping_vectors += [None]
            template_rects_full += [None]
            template_rects += [None]
            templates += [None]
            template_refs += [None]
            continue

        # a negative -x stepping vector means that the sample image is to the right of the reference image
        # a negative -y stepping vector means that the sample image is below the reference image
        stepping_vectors += [stepping_vector_px]
        template_rects_full += [template_rect_full]

        # scale down the intersection template so we have a search space:
        # It's a balance between search space (where you can slide the template around)
        # and specificity (bigger template will have a chance of encapsulating more features)
        SEARCH_SCALE = 0.80  # 0.8 worked on the AW set
        if True:
            template_rect = template_rect_full.scale(SEARCH_SCALE)
        else:
            # turns out a smaller, square template works better in general?
            template_rect = template_rect_full.scale(SEARCH_SCALE).to_square()
            # heuristic: slide the template "up" just a little bit because we generally have
            # more overlap toward the edge of the frame
            up_offset = template_rect.tl.y / 2
            template_rect = template_rect.translate(Point(0, -up_offset))

        template_rects += [template_rect]
        template = moving_img[
            round(template_rect.tl.y) : round(template_rect.br.y),
            round(template_rect.tl.x) : round(template_rect.br.x)
        ].copy()
        templates += [template]
        # trace the template's extraction point back to the moving rectangle's origin
        scale_offset = template_rect.tl - template_rect_full.tl
        template_ref = (-stepping_vector_px).clamp_zero() + scale_offset
        template_refs += [template_ref]
        no_overlaps = False

    #### handle the case that we're just totally out of the ball park
    if no_overlaps:
        self.schema.store_auto_align_result(moving_layer, None, False, solutions=0)
        logging.warning("No overlap between any reference and moving frame")
        err = np.zeros((600, 1000), dtype=np.uint8)
        cv2.putText(
            err, f"NO OVERLAP: {ref_meta['x']}, {ref_meta['y']} : {moving_meta['x']}, {moving_meta['y']} / {stepping_vector_px}",
            org=(50, 100),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        overview = np.hstack(pad_images_to_same_size(
            (
                cv2.resize(ref_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                cv2.resize(moving_img, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
            )
        ))
        cv2.imshow('before/after', np.vstack(
            pad_images_to_same_size((err, overview))
        ))
        cv2.waitKey() # pause because no delay is specified
        return removed

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
    manual_selection = None # record which ref image index is picked by the user
    recompute_all = False # set when toggling between ref images; we want to recompute all the views during this transition
    while mode is not None:
        msg = ''
        ##### Compute auto-alignment scores
        if mode == 'AUTO' or mode == 'TEMPLATE':
            best_mse = 1e100
            best_match = (0, 0)

            ref_laplacians = []
            contours = []
            hierarchies = []
            match_pts = []
            results = []
            solutions = []
            if Schema.FILTER_WINDOW > 0:
                moving_norm = cv2.GaussianBlur(moving_img, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                moving_norm = moving_norm.astype(np.float32)
            else:
                moving_norm = cv2.normalize(moving_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            moving_laplacian = cv2.Laplacian(moving_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
            # iterate over reference images
            for index, (ref_img, ref_meta, ref_t) in enumerate(ref_data):
                if templates[index] is not None \
                    and ((manual_selection is None) \
                         # don't process ref images that aren't selected
                         or (index == manual_selection)\
                         or recompute_all
                    ):
                    # blur over the range of one wavelength
                    if Schema.FILTER_WINDOW > 0:
                        ref_norm = cv2.GaussianBlur(ref_img, (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                        template_norm = cv2.GaussianBlur(templates[index], (Schema.FILTER_WINDOW, Schema.FILTER_WINDOW), 0)
                        ref_norm = ref_norm.astype(np.float32)
                        template_norm = template_norm.astype(np.float32)
                    else: # normalize instead of filter
                        ref_norm = cv2.normalize(ref_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        template_norm = cv2.normalize(templates[index], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # find edges
                    ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
                    ref_laplacians += [ref_laplacian]
                    template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

                    # apply template matching. If the ref image and moving image are "perfectly aligned",
                    # the value of `match_pt` should be equal to `template_ref`
                    # i.e. alignment criteria: match_pt - template_ref = (0, 0)
                    if True:
                        METHOD = cv2.TM_CCOEFF  # convolutional matching
                        res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        match_pt = max_loc
                        match_pts += [match_pt]
                        res_8u = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        results += [res_8u]
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
                    contour, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours += [contour]
                    hierarchies += [hierarchy]

                    has_single_solution = True
                    score = None
                    num_solns = len(hierarchy[0])
                    for i, c in enumerate(contour):
                        if hierarchy[0][i][3] == -1 and num_solns < MAX_SOLUTIONS:
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
                                logging.debug(f"{match_pt} of {i} is contained within a donut-shaped region. Suspect blurring error!")
                                has_single_solution = False
                    if score is not None:
                        logging.info(f"{i}: {score:0.1f} ({num_solns})")
                    else:
                        logging.info(f"{i}: No solution")
                    solutions += [(has_single_solution, score, num_solns)]
                else:
                    ref_laplacians += [None]
                    match_pts += [None]
                    results += [None]
                    contours += [None]
                    hierarchies += [None]
                    solutions += [(None, None, None)]
            recompute_all = False

        if manual_selection is None:
            score = None
            picked = 0
            for i, (ss, soln_score, num_solns) in enumerate(solutions):
                if ss is None:
                    continue # skip this selection because it's not valid
                if score is None:
                    picked = i
                    score = soln_score
                    has_single_solution = ss
                    solution_count = num_solns
                else:
                    if soln_score is not None:
                        if soln_score < score:
                            score = soln_score
                            has_single_solution = ss
                            picked = i
                            solution_count = num_solns
        else:
            picked = manual_selection
            (has_single_solution, score, solution_count) = solutions[picked]
        logging.info(f"picked: {picked}")
        match_pt = match_pts[picked]

        ##### Compute the putative adjustment vector
        adjustment_vector_px = Point(
            match_pts[picked][0] - template_refs[picked][0],
            match_pts[picked][1] - template_refs[picked][1]
        )

        #### Decide if user feedback is needed
        # use the contours and the matched point to measure the quality of the template match
        if retrench:
            logging.info("Manual QC flagged")
            mode = 'TEMPLATE'
            retrench = False # so we don't keep triggering this in the loop
        else:
            if score is not None and has_single_solution and mode == 'AUTO' and score < FAILING_SCORE:
                if full_review:
                    mode = 'TEMPLATE'
                else:
                    pass # this is the "everything worked" case
            elif mode == 'AUTO': # guess an initial mode for fix-up
                if abs(adjustment_vector_px.x) > X_REVIEW_THRESH or abs(adjustment_vector_px.y) > Y_REVIEW_THRESH:
                    mode = 'MOVE'
                    if score is None:
                        checked_score = -1.0
                    else:
                        checked_score = score
                    msg = f'LARGE OFFSET: {adjustment_vector_px} score: {checked_score:0.1f}'
                if score is not None and has_single_solution and score >= FAILING_SCORE:
                    logging.warning(f"Manual quality check: score {score} >= {FAILING_SCORE}")
                    mode = 'MOVE'
                    msg = f'QUALITY CHECK: {score:0.1f}'
                else:
                    logging.warning(f"Could not find unique solution: {solution_count} matches found")
                    mode = 'TEMPLATE'
        if mode == 'TEMPLATE':
            if score == None:
                checked_score = -1.0
            else:
                checked_score = score
            if full_review:
                msg = f'REVIEW: {checked_score:0.1f} of {solution_count}'
            else:
                msg = f'TEMPLATE: {checked_score:0.1f} of {solution_count}'

        ##### Render user feedback
        ref_img, ref_meta, ref_t = ref_data[picked]
        if results[picked] is not None and contours[picked] is not None:
            cv2.drawContours(results[picked], contours[picked], -1, (0,255,0), 1)
            cv2.imshow('contours', cv2.resize(results[picked], None, None, 0.5, 0.5))
        # compute after vector without storing the result
        after_vector_px = Point(int(
            - round(adjustment_vector_px.x) +
            round((
                ref_meta['x'] * 1000 + ref_t['offset'][0]
                - (moving_meta['x'] * 1000 + moving_t['offset'][0])
            ) * Schema.PIX_PER_UM)),
            int(- round(adjustment_vector_px.y) +
            round((
                ref_meta['y'] * 1000 + ref_t['offset'][1]
                - (moving_meta['y'] * 1000 + moving_t['offset'][1])
            ) * Schema.PIX_PER_UM))
        )
        ref_overlap = full_frame.intersection(
            full_frame.translate(Point(0, 0) - after_vector_px)
        )
        moving_overlap = full_frame.intersection(
            full_frame.translate(after_vector_px)
        )
        if ref_overlap is None or moving_overlap is None:
            after = np.zeros((500, 500), dtype=np.uint8)
        else:
            after = np.hstack(pad_images_to_same_size(
                (
                    # int() not round() used because python's "banker's rounding" sucks.
                    cv2.resize(ref_img[
                        int(ref_overlap.tl.y) : int(ref_overlap.br.y),
                        int(ref_overlap.tl.x) : int(ref_overlap.br.x)
                    ], None, None, PREVIEW_SCALE * SEARCH_SCALE, PREVIEW_SCALE * SEARCH_SCALE),
                    cv2.resize(moving_img[
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
                        cv2.resize(ref_laplacians[picked], None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                        cv2.resize(moving_laplacian, None, None, PREVIEW_SCALE / 2, PREVIEW_SCALE / 2),
                    )
                )),
                None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        cv2.putText(
            overview, 'lap(REF)',
            org=(25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        cv2.putText(
            overview, 'lap(SAMPLE) (unaligned)',
            org=(overview.shape[1] // 2 + 25, 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )
        before_after = np.vstack(
            pad_images_to_same_size(
                (cv2.resize(templates[picked], None, None, PREVIEW_SCALE, PREVIEW_SCALE),
                after, overview)
            )
        )
        cv2.imshow('before/after', before_after)

        ref_canvas = np.zeros(
            (ref_img.shape[0] + ceil(abs(after_vector_px.y)),
             ref_img.shape[1] + ceil(abs(after_vector_px.x))
            ), dtype=np.uint8
        )
        moving_canvas = np.zeros(
            (ref_img.shape[0] + ceil(abs(after_vector_px.y)),
             ref_img.shape[1] + ceil(abs(after_vector_px.x))
            ), dtype=np.uint8
        )
        ref_orig = Point(0, 0)
        moving_orig = Point(0, 0)
        rh, rw = ref_img.shape
        if after_vector_px.x >= 0:
            ref_orig.x = round(after_vector_px.x)
            moving_orig.x = 0
        else:
            ref_orig.x = 0
            moving_orig.x = round(-after_vector_px.x)
        if after_vector_px.y >= 0:
            ref_orig.y = round(after_vector_px.y)
            moving_orig.y = 0
        else:
            ref_orig.y = 0
            moving_orig.y = round(-after_vector_px.y)
        ref_canvas[
            ref_orig.y : ref_orig.y + rh,
            ref_orig.x : ref_orig.x + rw
        ] = ref_img
        moving_canvas[
            moving_orig.y : moving_orig.y + rh,
            moving_orig.x : moving_orig.x + rw
        ] = moving_img
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

        ##### Handle UI cases
        if score is not None and has_single_solution and mode == 'AUTO' and score < FAILING_SCORE: # passing case
            cv2.waitKey(10)
            mode = None # this causes the stitch loop to exit and store the alignment value
        else: # failure cases
            if mode == 'MOVE':
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
                corr_f32 = cv2.normalize(corr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                (_retval, corr_f32) = cv2.threshold(corr_f32, 0.8, 0.8, cv2.THRESH_TRUNC) # toss out extreme outliers (e.g. bad pixels)
                corr_f32 = cv2.normalize(corr_f32, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # corr_u8 = cv2.normalize(corr_f32, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                err = np.sum(corr**2)
                h, w = ref_laplacian.shape
                mse = err / (float(h*w))
                log_mse = round(log10(mse), 4)
                if log_mse <= best_mse:
                    best_mse = log_mse
                    mse_hint = 'best'
                    best_match = match_pt
                else:
                    mse_hint = ''
                logging.info(f"MSE: {log_mse} {mse_hint}")
                cv2.putText(
                    corr_f32, f"MSE: {log_mse} {mse_hint}",
                    org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA
                )
                cv2.imshow("Find minimum MSE", cv2.resize(corr_f32, None, None, 0.5, 0.5))

                # get user feedback and adjust the match_pt accordingly
                logging.warning(f"MOVE IMAGE: 'wasd' to move, space to accept, 1 to toggle to template mode, x to remove from database, y to snap to best point, Y to zero match pt")
                key = cv2.waitKey()
                COARSE_MOVE = 20
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
                    # coarse movement
                    elif key_char == '<' or key_char == 'W':
                        match_pt = (match_pt[0] + 0, match_pt[1] - COARSE_MOVE)
                    elif key_char == 'A':
                        match_pt = (match_pt[0] - COARSE_MOVE, match_pt[1] + 0)
                    elif key_char == 'E' or key_char == 'D':
                        match_pt = (match_pt[0] + COARSE_MOVE, match_pt[1] + 0)
                    elif key_char == 'O' or key_char == 'S':
                        match_pt = (match_pt[0] + 0, match_pt[1] + COARSE_MOVE)
                    # reset match point to what a perfect machine would yield
                    elif key_char == 'Y' or key_char == 'T':
                        match_pt = Point(template_ref[0], template_ref[1]) + \
                            Point(
                                round(
                                    (ref_t['offset'][0] - moving_t['offset'][0]) * Schema.PIX_PER_UM),
                                round(
                                    (ref_t['offset'][1] - moving_t['offset'][1]) * Schema.PIX_PER_UM)
                            )
                        best_mse = 10e100 # reset the MSE score since we're in a totally different regime
                    elif key_char == 'y' or key_char == 't':
                        match_pt = best_match
                    elif key_char == ' ': # this accepts the current alignment
                        score = FAILING_SCORE - 1
                        mode = None
                        has_single_solution = True # force this to true
                    elif key_char == 'x': # this rejects the alignment
                        self.schema.remove_tile(moving_layer)
                        logging.info(f"Removing tile {moving_layer} from the database")
                        removed = True
                        mode = None
                    elif key_char == '1':
                        mode = 'TEMPLATE'
                    elif key_char == '\t': # toggle to another solution
                        while True:
                            picked = (picked + 1) % len(ref_data)
                            if templates[picked] is not None: # make sure the picked index is not invalid
                                break
                        manual_selection = picked
                        recompute_all = True
                        logging.info(f"Toggling to index {picked}")
                    else:
                        logging.debug(f"Unhandled key: {key_char}, ignoring")
                    manual_selection = picked # to avoid images from changing res during shifting
                    match_pts[picked] = match_pt
            elif mode == 'TEMPLATE':
                if score is not None and has_single_solution and score < FAILING_SCORE:
                    hint = 'PASS'
                else:
                    hint = 'failing'
                logging.info(f"score {score}, solutions: {solution_count} ({hint})")
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
                            picked = (picked + 1) % len(ref_data)
                            if templates[picked] is not None: # make sure the picked index is not invalid
                                break
                        manual_selection = picked
                        recompute_all = True
                        logging.info(f"Toggling to index {picked}")
                    else:
                        continue
                    manual_selection = picked # to avoid images from changing res during shifting
                    if template_shift is not None:
                        template_rects[picked] = template_rects[picked].saturating_translate(template_shift, full_frame)
                        templates[picked] = moving_img[
                            round(template_rects[picked].tl.y) : round(template_rects[picked].br.y),
                            round(template_rects[picked].tl.x) : round(template_rects[picked].br.x)
                        ].copy()
                        # trace the template's extraction point back to the moving rectangle's origin
                        scale_offset = template_rects[picked].tl - template_rects_full[picked].tl
                        template_refs[picked] = (-stepping_vectors[picked]).clamp_zero() + scale_offset

        # store the result if the mode is set to None, and the schema still contains the moving layer.
        if mode == None and self.schema.contains_layer(moving_layer):
            # Exit loop: store the stitch result
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
                solutions=solution_count
            )
            check_t = self.schema.schema['tiles'][str(moving_layer)]
            logging.info(f"after adjustment: {check_t['offset'][0]:0.2f}, {check_t['offset'][1]:0.2f} score: {score} candidates: {solution_count}")
    return removed

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
    for x in x_indices:
        for y in y_indices:
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
                (_ref_layer, ref_t) = self.schema.get_tile_by_coordinate(cur_tile)
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
                    ref_img = self.schema.get_image_from_tile(ref_t)
                    moving_img = self.schema.get_image_from_tile(moving_t)
                    logging.info(f"Stitching {cur_tile} to {next_tile}")
                    if prev_x is not None: # add the left tile as an option for stitching (stitching goes left-to-right)
                        (_left_layer, left_t) = self.schema.get_tile_by_coordinate(Point(prev_x, y))
                        if left_t is not None:
                            left_img = self.schema.get_image_from_tile(left_t)
                            ref_data = [(ref_img, Schema.meta_from_tile(ref_t), ref_t),
                                        (left_img, Schema.meta_from_tile(left_t), left_t)]
                        else:
                            ref_data = [(ref_img, Schema.meta_from_tile(ref_t), ref_t)]
                    else:
                        ref_data = [(ref_img, Schema.meta_from_tile(ref_t), ref_t)]
                    removed = self.stitch_one_template(
                        ref_data,
                        moving_img, Schema.meta_from_tile(moving_t), moving_t, moving_layer,
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
