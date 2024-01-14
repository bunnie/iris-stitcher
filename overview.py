import numpy as np
from schema import Schema
import logging
from prims import Rect, Point
from utils import safe_image_broadcast
from math import ceil

from PyQt5.QtGui import QPixmap, QImage

import cv2
import platform
if platform.system() == 'Linux':
    import os
    envpath = '/home/bunnie/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

def redraw_overview(self, blend=True):
    sorted_tiles = self.schema.sorted_tiles()
    canvas = np.zeros((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)
    # ones indicate regions that need to be copied
    if blend:
        mask = np.ones((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)
    else:
        mask = None

    for (layer, tile) in sorted_tiles:
        metadata = Schema.meta_from_fname(tile['file_name'])
        (x, y) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        # move center coordinate to top left
        x -= Schema.X_RES / 2
        y -= Schema.Y_RES / 2

        img = self.schema.get_image_from_layer(layer)
        result = safe_image_broadcast(img, canvas, x, y, mask)
        if result is not None:
            canvas, mask = result

    self.overview = canvas
    self.rescale_overview()

# This only rescales from a cached copy, does not actually recompute anything.
def rescale_overview(self):
    w = self.lbl_overview.width()
    h = self.lbl_overview.height()
    (x_res, y_res) = (self.schema.max_res[0], self.schema.max_res[1])
    # constrain by height and aspect ratio
    scaled = cv2.resize(self.overview, (int(x_res * (h / y_res)), h))
    height, width = scaled.shape
    bytesPerLine = 1 * width
    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
    ))
    self.overview_actual_size = (width, height)
    self.overview_scaled = scaled.copy()

def update_selected_rect(self, update_tile=False):
    (layer, tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
    selected_image = self.schema.get_image_from_layer(layer)
    metadata = Schema.meta_from_tile(tile)
    logging.info(f"Selected layer {layer}: {metadata['x']}, {metadata['y']} nom, {tile['offset']} offset")

    # Refactor: work from the original, composite, then scale down.
    # (Originally: work on scaled copy. Problem: subpixel snapping causes image to shift.)
    (x_c, y_c) = self.um_to_pix_absolute(
        (float(metadata['x']) * 1000 + float(tile['offset'][0]),
        float(metadata['y']) * 1000 + float(tile['offset'][1]))
    )
    ui_overlay = self.overview.copy()

    # define the rectangle
    w = selected_image.shape[1]
    h = selected_image.shape[0]
    tl_x = int(x_c - w/2)
    tl_y = int(y_c - h/2)
    tl = (tl_x, tl_y)
    br = (tl_x + int(w), tl_y + int(h))

    # overlay the tile
    if update_tile:
        safe_image_broadcast(selected_image, ui_overlay, tl[0], tl[1])

    # draw the rectangle
    h_target = self.lbl_overview.height()
    (x_res, y_res) = (self.schema.max_res[0], self.schema.max_res[1])
    thickness = y_res / h_target # get a 1-pix line after rescaling
    cv2.rectangle(
        ui_overlay,
        tl,
        br,
        (255, 255, 255),
        thickness = ceil(thickness),
        lineType = cv2.LINE_4
    )

    # use the same height-driven rescale as in `rescale_overview()`
    # constrain by height and aspect ratio
    scaled = cv2.resize(ui_overlay, (int(x_res * (h_target / y_res)), h_target))
    height, width = scaled.shape
    bytesPerLine = 1 * width
    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
    ))

    # update the status bar output
    (layer, t) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
    if t is not None:
        md = Schema.meta_from_fname(t['file_name'])
        self.status_centroid_ui.setText(f"{md['x']:0.2f}, {md['y']:0.2f}")
        self.status_layer_ui.setText(f"{layer}")
        self.status_is_anchor.setChecked(layer == self.schema.anchor_layer_index())
        self.status_offset_ui.setText(f"{t['offset'][0]:0.2f}, {t['offset'][1]:0.2f}")
        self.status_score.setText(f"{t['score']:0.3f}")
        self.status_stitch_err.setText(f"{t['auto_error']}")
        if md['r'] >= 0:
            self.status_rev_ui.setText(f"{int(md['r'])}")
        else:
            self.status_rev_ui.setText("average")
        if 'f' in md:
            self.status_fit_metric_ui.setText(f"{md['f']:0.1f}")
        else:
            self.status_fit_metric_ui.setText("None")
        if 's' in md:
            self.status_score_metric_ui.setText(f"{md['s']}")
        else:
            self.status_score_metric_ui.setText("None")
        if 'v' in md:
            self.status_ratio_metric_ui.setText(f"{md['v']:0.3f}")
        else:
            self.status_ratio_metric_ui.setText("None")

def get_coords_in_range(self):
    if self.select_pt1 is None or self.select_pt2 is None:
        logging.warning("Set two selection points using the '1' and '2' keys before continuing")
        return None

    boundary = Rect(self.select_pt1, self.select_pt2)
    coords_in_range = []
    for coords in self.schema.coords_mm:
        c = Point(coords[0], coords[1])
        if boundary.intersects(c):
            coords_in_range += [coords]
    return coords_in_range

def preview_selection(self):
    if self.select_pt1 is None or self.select_pt2 is None:
        logging.warning("Set two selection points using the '1' and '2' keys before continuing")
        return

    w = (self.overview_actual_size[0] / self.schema.max_res[0]) * Schema.X_RES
    h = (self.overview_actual_size[1] / self.schema.max_res[1]) * Schema.Y_RES
    ui_overlay = np.zeros(self.overview_scaled.shape, self.overview_scaled.dtype)
    coords_in_range = self.get_coords_in_range()
    for coord in coords_in_range:
        (_layer, tile) = self.schema.get_tile_by_coordinate(coord)
        metadata = Schema.meta_from_tile(tile)
        (x_c, y_c) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        # define the rectangle
        x_c = (self.overview_actual_size[0] / self.schema.max_res[0]) * x_c
        y_c = (self.overview_actual_size[1] / self.schema.max_res[1]) * y_c
        tl_x = int(x_c - w/2)
        tl_y = int(y_c - h/2)
        tl = (tl_x, tl_y)
        br = (tl_x + int(w), tl_y + int(h))
        cv2.rectangle(
            ui_overlay,
            tl,
            br,
            (128, 128, 128),
            thickness = 1,
            lineType = cv2.LINE_4
        )
    composite = cv2.addWeighted(self.overview_scaled, 1.0, ui_overlay, 0.5, 0.0)

    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(composite.data, self.overview_scaled.shape[1], self.overview_scaled.shape[0], self.overview_scaled.shape[1],
                QImage.Format.Format_Grayscale8)
    ))

# ASSUME: tile is Schema.X_RES, Schema.Y_RES in resolution
def centroid_to_tile_bounding_rect_mm(self, centroid_mm):
    (x_mm, y_mm) = centroid_mm
    w_mm = (Schema.X_RES / Schema.PIX_PER_UM) / 1000
    h_mm = (Schema.Y_RES / Schema.PIX_PER_UM) / 1000

# compute a window that is `opening` wide that tries its best to center around
# `center`, but does not exceed [0, max)
def snap_range(self, x_off, w, max):
    assert max >= w, "window requested is wider than the maximum image resolution"
    # check if we have space on the left
    if x_off - w/2 >= 0:
        if x_off + w/2 <= max:
            return (int(x_off - w/2), int(x_off + w/2)) # window fits!
        else:
            return (int(max - w), max) # snap window to the right
    else:
        return (0, w) # snap window to the left

# checks that a value is between [0, max):
def check_res_bounds(self, x, max):
    if x < 0:
        print(f"Res check got {x} < 0", x)
        return 0
    elif x >= max:
        print(f"Res check got {x} >= {max}", x, max)
        return max - 1
    else:
        return x

def pix_to_um_absolute(self, pix, cur_res):
    (x, y) = pix
    (res_x, res_y) = cur_res
    return (
        x * (self.schema.max_res[0] / res_x) / Schema.PIX_PER_UM + self.schema.x_min_mm * 1000,
        y * (self.schema.max_res[1] / res_y) / Schema.PIX_PER_UM + self.schema.y_min_mm * 1000
    )
def um_to_pix_absolute(self, um):
    (x_um, y_um) = um
    return (
        int((x_um - self.schema.x_min_mm * 1000) * Schema.PIX_PER_UM),
        int((y_um - self.schema.y_min_mm * 1000) * Schema.PIX_PER_UM)
    )