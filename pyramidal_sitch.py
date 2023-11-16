from schema import Schema
from prims import Rect, Point, ROUNDING
import cv2
import numpy as np
import logging

# TODO recode this template
# using pyramidal downsampling
# src = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))

def stitch_one_pyramidal(self):
    (x_um, y_um) = self.roi_center_ums
    canvas_xres = Schema.X_RES * 3 + 2
    canvas_yres = Schema.Y_RES * 3 + 2
    canvas_center = (canvas_xres // 2, canvas_yres // 2)
    canvas_rect = Rect(
        Point(0, 0),
        Point(canvas_xres, canvas_yres)
    )

    # algorithm:
    # measure std deviation of the differences of laplacians and do a gradient descent.
    # First, extract the two tiles we're aligning: the reference tile, and the moving tile.
    ref_img = None
    moving_img = None
    for (layer, t, img) in self.schema.zoom_cache:
        meta = Schema.meta_from_tile(t)
        center_offset_px = (
            int((float(meta['x']) * 1000 + t['offset'][0] - x_um) * Schema.PIX_PER_UM),
            int((float(meta['y']) * 1000 + t['offset'][1] - y_um) * Schema.PIX_PER_UM)
        )
        x = center_offset_px[0] - Schema.X_RES // 2 + canvas_center[0]
        y = center_offset_px[1] - Schema.Y_RES // 2 + canvas_center[1]

        if layer == self.ref_layer:
            ref_img = img
            ref_bounds =  Rect(
                Point(x, y),
                Point(x + Schema.X_RES, y + Schema.Y_RES)
            )
        elif layer == self.selected_layer:
            # moving_bounds computed in the main loop
            moving_img = img
            moving_meta = meta
            moving_t = t

    if ref_img is not None and moving_img is not None:
        SEARCH_EXTENT_PX = 30 # pixels in each direction. about +/-3 microns or so in actual size, so a 6 um^2 total search area.
        SEARCH_REGION_PX = 512 # dimension of the fast search region, in pixels
        SEARCH_TOLERANCE_PX = 2 # limit of search refinement - set at 2px for 20x lens because we are beyond quantum limit
        DEBUG = False
        extra_offset_y_px = -SEARCH_EXTENT_PX
        extra_offset_x_px = 0 # Y-search along the nominal centerline, then search X extent ("T-shaped" search)
        align_scores_y = {} # search in Y first. Scores are {pix_offset : score} entries
        align_scores_x = {} # then search in X
        state = 'SEARCH_VERT'
        # DONE means we found a minima
        # ABORT means we couldn't find one

        from datetime import datetime
        start = datetime.now()
        print(f"starting offset: {moving_t['offset'][0]}, {moving_t['offset'][1]}")
        full_frame = False
        full_frame_recompute = False
        check_mses = []

        while state != 'DONE' and state != 'ABORT':
            center_offset_px = (
                int((float(moving_meta['x']) * 1000 + moving_t['offset'][0] - x_um) * Schema.PIX_PER_UM) + extra_offset_x_px,
                int((float(moving_meta['y']) * 1000 + moving_t['offset'][1] - y_um) * Schema.PIX_PER_UM) + extra_offset_y_px
            )
            # print(f"{center_offset_px} / {extra_offset_x_px}, {extra_offset_y_px}")
            x = center_offset_px[0] - Schema.X_RES // 2 + canvas_center[0]
            y = center_offset_px[1] - Schema.Y_RES // 2 + canvas_center[1]
            moving_bounds =  Rect(
                Point(x, y),
                Point(x + Schema.X_RES, y + Schema.Y_RES)
            )

            roi_bounds = ref_bounds.intersection(moving_bounds)
            # narrow down the search region if the ROI is larger than the specified search region
            if roi_bounds.width() >= SEARCH_REGION_PX and roi_bounds.height() >= SEARCH_REGION_PX \
            and not full_frame:
                subrect = Rect(
                    Point(0, 0),
                    Point(SEARCH_REGION_PX, SEARCH_REGION_PX)
                )
                subrect = subrect.translate(
                    roi_bounds.tl +
                    Point(
                        roi_bounds.width() // 2 - subrect.width() // 2,
                        roi_bounds.height() // 2 - subrect.height() // 2
                    )
                )
                roi_bounds = roi_bounds.intersection(subrect)

            # print(roi_bounds)
            if roi_bounds is not None:
                # Compute the intersecting pixels only between the two images, without copying
                ref_clip = canvas_rect.intersection(roi_bounds)
                ref_roi_rect = ref_clip.translate(Point(0, 0) - ref_clip.tl) # move rectangle to 0,0 reference frame
                ref_roi_rect = ref_roi_rect.translate(roi_bounds.tl - ref_bounds.tl) # apply ref vs roi bounds offset
                ref_roi = ref_img[
                    ref_roi_rect.tl.y : ref_roi_rect.br.y,
                    ref_roi_rect.tl.x : ref_roi_rect.br.x
                ]

                moving_clip = canvas_rect.intersection(moving_bounds).intersection(roi_bounds)
                moving_roi_rect = moving_clip.translate(Point(0, 0) - moving_clip.tl)
                moving_roi_rect = moving_roi_rect.translate(roi_bounds.tl - moving_bounds.tl)
                moving_roi = moving_img[
                    moving_roi_rect.tl.y : moving_roi_rect.br.y,
                    moving_roi_rect.tl.x : moving_roi_rect.br.x
                ]

                # now find the difference
                ref_norm = cv2.normalize(ref_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                moving_norm = cv2.normalize(moving_roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                ## difference of laplacians (removes effect of lighting gradient)
                # 15 is too small, 31 works, 27 also seems fine? This may need to be a tunable param based on the exact chip we're imaging, too...
                # but overall this should be > than pixels/um * 1.05um, i.e., the wavelength of of the light which defines the minimum
                # feature we could even reasonably have contrast over. recall 1.05um is wavelength of light.
                # pixels/um * 1.05um as of the initial build is 10, so, 27 would be capturing an area of about 2.7 wavelengths.
                ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
                moving_laplacian = cv2.Laplacian(moving_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

                # corr = moving_laplacian - ref_laplacian
                h, w = ref_laplacian.shape
                corr = cv2.subtract(moving_laplacian, ref_laplacian)
                corr_u8 = cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if DEBUG:
                    cv2.imshow('progress', corr_u8)
                err = np.sum(corr**2)
                mse = err / (float(h*w))
                # now evaluate if we've reached a minima in our particular search direction, or if we should try searching the other way
                if state == 'SEARCH_VERT':
                    if DEBUG:
                        cv2.waitKey(30)
                    align_scores_y[extra_offset_y_px] = mse #np.std(corr)
                    if extra_offset_y_px == SEARCH_EXTENT_PX:
                        s = np.array(sorted(align_scores_y.items(), key=lambda x: x[0]))  # sort by pixel offset
                        min_row = s[np.argmin(s[:, 1])] # extract the row with the smallest mse value
                        extra_offset_y_px = int(min_row[0])
                        state = 'SHOW_VERT'
                        # extra_offset_x_px = -SEARCH_EXTENT_PX
                    else:
                        extra_offset_y_px += 1
                elif state == 'SHOW_VERT':
                    if DEBUG:
                        print(f"vertical alignment: f{extra_offset_y_px}")
                        cv2.waitKey()
                    extra_offset_x_px = -SEARCH_EXTENT_PX
                    state = 'SEARCH_HORIZ'
                elif state == 'SEARCH_HORIZ':
                    if DEBUG:
                        cv2.waitKey(30)
                    align_scores_x[extra_offset_x_px] = mse #np.std(corr)
                    if extra_offset_x_px == SEARCH_EXTENT_PX:
                        s = np.array(sorted(align_scores_x.items(), key=lambda x: x[0]))
                        min_row = s[np.argmin(s[:, 1])]
                        extra_offset_x_px = int(min_row[0])
                        state = 'SHOW_HORIZ'
                    else:
                        extra_offset_x_px += 1
                elif state == 'SHOW_HORIZ':
                    fast_alignment_pt = Point(extra_offset_x_px, extra_offset_y_px)
                    full_frame = True
                    if DEBUG:
                        print("showing final pick")
                        cv2.waitKey()
                    if full_frame_recompute:
                        print(f"Slowly recomputed alignment: {fast_alignment_pt}, score: {mse}")
                        state = 'DONE'
                    else:
                        print(f"Fast alignment: {fast_alignment_pt}, score: {mse}")
                        state = 'CHECK_PICK'
                elif state == 'CHECK_PICK':
                    check_mses += [mse] # first insertion: our "picked" MSE is at index 0
                    extra_offset_x_px = fast_alignment_pt.x + SEARCH_TOLERANCE_PX
                    state = 'CHECK_X+'
                elif state == 'CHECK_X+':
                    check_mses += [mse]
                    extra_offset_x_px = fast_alignment_pt.x - SEARCH_TOLERANCE_PX
                    state = 'CHECK_X-'
                elif state == 'CHECK_X-':
                    check_mses += [mse]
                    extra_offset_x_px = fast_alignment_pt.x
                    extra_offset_y_px = fast_alignment_pt.y + SEARCH_TOLERANCE_PX
                    state = 'CHECK_Y+'
                elif state == 'CHECK_Y+':
                    check_mses += [mse]
                    extra_offset_y_px = fast_alignment_pt.y - SEARCH_TOLERANCE_PX
                    state = 'FINAL_CHECK'
                elif state == 'FINAL_CHECK':
                    print(f"checked mses: {check_mses}")
                    if check_mses[0] != min(check_mses):
                        logging.warning("Fast search did not yield an optimal result! Re-doing with a slow, full-frame search.")
                        full_frame_recompute = True
                        extra_offset_y_px = -SEARCH_EXTENT_PX
                        extra_offset_x_px = 0 # Y-search along the nominal centerline, then search X extent ("T-shaped" search)
                        align_scores_y = {} # search in Y first. Scores are {pix_offset : score} entries
                        align_scores_x = {} # then search in X
                        state = 'SEARCH_VERT'
                    else:
                        state = 'DONE'
            else:
                state = 'ABORT'
                logging.warning("Regions lost overlap during auto-stitching, aborting!")

        #import pprint
        #print("x scores:")
        #pprint.pprint(align_scores_x)
        #print("y scores:")
        #pprint.pprint(align_scores_y)
        print("2x {} search done in {}".format(SEARCH_EXTENT_PX, datetime.now() - start))
        print(f"minima at: {fast_alignment_pt}")
        print(f"before adjustment: {moving_t['offset'][0]},{moving_t['offset'][1]}")
        # now update the offsets to reflect this
        self.schema.adjust_offset(
            self.selected_layer,
            fast_alignment_pt.x / Schema.PIX_PER_UM,
            fast_alignment_pt.y / Schema.PIX_PER_UM
        )
        check_t = self.schema.schema['tiles'][str(self.selected_layer)]
        print(f"after adjustment: {check_t['offset'][0]},{check_t['offset'][1]}")

