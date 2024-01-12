import numpy as np
from schema import Schema
import logging
from prims import Rect, Point
from utils import safe_image_broadcast

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
    (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
    metadata = Schema.meta_from_tile(tile)
    (x_c, y_c) = self.um_to_pix_absolute(
        (float(metadata['x']) * 1000 + float(tile['offset'][0]),
        float(metadata['y']) * 1000 + float(tile['offset'][1]))
    )
    ui_overlay = np.zeros(self.overview_scaled.shape, self.overview_scaled.dtype)

    # define the rectangle
    w = (self.overview_actual_size[0] / self.schema.max_res[0]) * self.cached_image.shape[1]
    h = (self.overview_actual_size[1] / self.schema.max_res[1]) * self.cached_image.shape[0]
    x_c = (self.overview_actual_size[0] / self.schema.max_res[0]) * x_c
    y_c = (self.overview_actual_size[1] / self.schema.max_res[1]) * y_c
    tl_x = int(x_c - w/2)
    tl_y = int(y_c - h/2)
    tl = (tl_x, tl_y)
    br = (tl_x + int(w), tl_y + int(h))

    # overlay the tile
    # constrain resize by the same height and aspect ratio used to generate the overall image
    scaled_tile = cv2.resize(
        self.cached_image,
        (int(w), int(h))
    )
    if update_tile:
        ui_overlay[tl[1]:tl[1] + int(h), tl[0]:tl[0] + int(w)] = scaled_tile

    # draw the rectangle
    cv2.rectangle(
        ui_overlay,
        tl,
        br,
        (128, 128, 128),
        thickness = 1,
        lineType = cv2.LINE_4
    )

    if update_tile:
        # just overlay, don't blend
        composite = self.overview_scaled.copy()
        composite[tl[1]:tl[1] + int(h), tl[0]:tl[0] + int(w)] = ui_overlay[tl[1]:tl[1] + int(h), tl[0]:tl[0] + int(w)]
    else:
        composite = cv2.addWeighted(self.overview_scaled, 1.0, ui_overlay, 0.5, 0.0)

    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(composite.data, self.overview_scaled.shape[1], self.overview_scaled.shape[0], self.overview_scaled.shape[1],
                QImage.Format.Format_Grayscale8)
    ))

    # update the status bar output
    (layer, t) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
    if t is not None:
        md = Schema.meta_from_fname(t['file_name'])
        self.status_centroid_ui.setText(f"{md['x']:0.2f}, {md['y']:0.2f}")
        self.status_layer_ui.setText(f"{layer}")
        self.status_is_anchor.setChecked(layer == self.schema.anchor_layer_index())
        self.status_offset_ui.setText(f"{t['offset'][0]:0.2f}, {t['offset'][1]:0.2f}")
        self.status_score.setText(f"{t['score']:0.3f}")
        self.status_stitch_err.setText(f"{t['auto_error']}")
        if md['r'] >= 0:
            self.status_rev_ui.setText(f"{md['r']}")
        else:
            self.status_rev_ui.setText("average")

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

    w = (self.overview_actual_size[0] / self.schema.max_res[0]) * self.cached_image.shape[1]
    h = (self.overview_actual_size[1] / self.schema.max_res[1]) * self.cached_image.shape[0]
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