import numpy as np
import cv2
from schema import Schema
import logging
from prims import Rect, Point

WINDOW_SIZE_X = 2000
WINDOW_SIZE_Y = 2000

def on_cv_zoom(self, raw_value):
    # raw_value goes from 0-100.
    # 0 = 0.1x zoom. 20 = 1.0x zoom. 100 = 10.0x zoom.
    if raw_value >= 20:
        self.zoom_scale = ((10.0 - 1.0) / (100 - 20)) * raw_value - 1.25
    else:
        self.zoom_scale = ((1.0 - 0.1) / (20 - 0)) * raw_value + 0.1

    img = self.get_centered_and_scaled_image()
    cv2.imshow('zoom', cv2.resize(img, None, None, self.zoom_scale, self.zoom_scale))

def get_centered_and_scaled_image(self):
    # ensure that the fullres dataset is pulled in. Warning: could take a while.
    if self.overview_fullres is None:
        self.redraw_overview(blend=False, force_full_res=True)

    (x_c, y_c) = self.um_to_pix_absolute(self.roi_center_ums)
    # scale of 2.0 means we are zooming in by 2x; 0.5 means we are zooming out by 2x

    # define the "ideal" rectangle that is the range of pixels we want to encompass
    # the entire zoom set
    selection_unchecked = Rect(
        Point(int(x_c - (WINDOW_SIZE_X / self.zoom_scale) / 2), int(y_c - (WINDOW_SIZE_Y / self.zoom_scale) / 2)),
        Point(int(x_c + (WINDOW_SIZE_X / self.zoom_scale) / 2), int(y_c + (WINDOW_SIZE_Y / self.zoom_scale) / 2)),
    )
    # overall size of the target image
    canvas_r = Rect(Point(0, 0), Point(self.overview_fullres.shape[1], self.overview_fullres.shape[0]))
    # intersect the unchecked with the canvas to get a checked selection
    selection = canvas_r.intersection(selection_unchecked)
    if selection is None:
        logging.warning("Error computing selection in zoom, ignoring request")
        return

    img = self.overview_fullres[
        selection.tl.y : selection.tl.y + selection.height(),
        selection.tl.x : selection.tl.x + selection.width()
    ]
    return img

def update_zoom_window(self):
    img = self.get_centered_and_scaled_image()
    cv2.imshow('zoom', cv2.resize(img, None, None, self.zoom_scale, self.zoom_scale))
    if not self.trackbar_created:
        cv2.createTrackbar('scale', 'zoom', 20, 100, self.on_cv_zoom)
        self.trackbar_created = True
