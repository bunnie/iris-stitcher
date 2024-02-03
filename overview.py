from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QRadioButton

import cv2
import platform
if platform.system() == 'Linux':
    import os
    envpath = '/home/bunnie/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import logging
import json
from pathlib import Path

from schema import Schema
from prims import Rect, Point
from utils import safe_image_broadcast
from math import ceil, sqrt

from config import *
from progressbar.bar import ProgressBar

# This generates a black-and-white only full resolution overview, suitable for saving to files.
def generate_fullres_overview(self, blend=True):
    sorted_tiles = self.schema.sorted_tiles()
    canvas = np.zeros((int(self.schema.max_res[1]), int(self.schema.max_res[0])), dtype=np.uint8)
    # ones indicate regions that need to be copied
    if blend:
        mask = np.ones((int(self.schema.max_res[1]), int(self.schema.max_res[0])), dtype=np.uint8)
    else:
        mask = None

    description='full res'
    progress = ProgressBar(min_value=0, max_value=len(sorted_tiles), prefix=f'Loading {description} tiles... ').start()
    for (index, (layer, tile)) in enumerate(sorted_tiles):
        metadata = Schema.meta_from_fname(tile['file_name'])
        (x, y) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        # move center coordinate to top left
        x -= X_RES / 2
        y -= Y_RES / 2

        img = self.schema.get_image_from_layer(layer, thumb=False).copy()
        result = safe_image_broadcast(img, canvas, x, y, mask, 1.0)
        if result is not None:
            canvas, mask = result
        progress.update(index)
    progress.finish()

    self.overview_fullres = canvas

# This generates a thumbnailed color overview, suitable for screen display. Also does overlay processing.
def redraw_overview(self, blend=True):
    scale = THUMB_SCALE
    sorted_tiles = self.schema.sorted_tiles()
    canvas = np.zeros((int(self.schema.max_res[1] * scale), int(self.schema.max_res[0] * scale), 3), dtype=np.uint8)
    # ones indicate regions that need to be copied
    if blend:
        mask = np.ones((int(self.schema.max_res[1] * scale), int(self.schema.max_res[0] * scale)), dtype=np.uint8)
    else:
        mask = None

    description='thumbnail'
    progress = ProgressBar(min_value=0, max_value=len(sorted_tiles), prefix=f'Loading {description} tiles... ').start()
    for (index, (layer, tile)) in enumerate(sorted_tiles):
        metadata = Schema.meta_from_fname(tile['file_name'])
        (x, y) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        # move center coordinate to top left
        x -= X_RES / 2
        y -= Y_RES / 2

        img = self.schema.get_image_from_layer(layer, thumb=True).copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if tile['auto_error'] == 'true':
            # "x-out" the tile as being flagged for manual review
            cv2.line(
                img,
                (0, 0),
                (img.shape[1], img.shape[0]),
                (255, 255, 255),
                50,
                lineType=cv2.LINE_AA
            )
            cv2.line(
                img,
                (img.shape[1], 0),
                (0, img.shape[0]),
                (255, 255, 255),
                50,
                lineType=cv2.LINE_AA
            )
        if self.layer_dist_dict is not None and layer in self.layer_dist_dict:
            dist = self.layer_dist_dict[layer]
            overlay = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(
                overlay,
                (0, 0),
                (img.shape[1], img.shape[0]),
                (int(dist * 0xFD), int(dist * 0xB0), int(dist * 0xC0)),
                -1
            )
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        result = safe_image_broadcast(img, canvas, x, y, mask, scale)
        if result is not None:
            canvas, mask = result
        progress.update(index)
    progress.finish()

    self.overview = canvas
    self.rescale_overview()
    if self.show_selection:
        self.preview_selection()

# This only rescales from a cached copy, does not actually recompute anything.
def rescale_overview(self):
    w = self.lbl_overview.width()
    h = self.lbl_overview.height()
    (y_res, x_res, _planes) = self.overview.shape
    # constrain by height and aspect ratio
    scaled = cv2.resize(self.overview, (int(x_res * (h / y_res)), h))
    height, width, planes = scaled.shape
    bytesPerLine = planes * width
    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    ))
    self.overview_actual_size = (width, height)
    self.overview_scaled = scaled.copy()

def update_selected_rect(self, update_tile=False):
    # Extract the list of intersecting tiles and update the UI
    closet_tiles = self.schema.get_intersecting_tiles((self.roi_center_ums[0] / 1000, self.roi_center_ums[1] / 1000))
    # clear all widgets from the vbox layout
    while self.status_layer_select_layout.count():
        child = self.status_layer_select_layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    first = True
    for (layer, t) in closet_tiles:
        md = Schema.meta_from_fname(t['file_name'])
        t_center = Point(float(md['x'] + t['offset'][0] / 1000), float(md['y'] + t['offset'][1] / 1000))
        b = QRadioButton(str(layer) + f': {t_center[0]:0.3f},{t_center[1]:0.3f}')
        if first:
            b.setChecked(True)
            first = False
        self.status_layer_select_layout.addWidget(b)
    # TODO: add routines to connect radio button actions to something that acts on it.

    # Draw the UI assuming the closest is the selected.
    (layer, tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
    selected_image = self.schema.get_image_from_layer(layer, thumb=True)
    metadata = Schema.meta_from_tile(tile)
    logging.info(f"Selected layer {layer}: {metadata['x']}, {metadata['y']} nom, {tile['offset']} offset")

    # Refactor: work from the original, composite, then scale down.
    # (Originally: work on scaled copy. Problem: subpixel snapping causes image to shift.)
    (x_c, y_c) = self.um_to_pix_absolute(
        (float(metadata['x']) * 1000 + float(tile['offset'][0]),
        float(metadata['y']) * 1000 + float(tile['offset'][1]))
    )
    ui_overlay = self.overview.copy()

    # x/y coords to safe_image_broadcast are unscaled
    w = selected_image.shape[1] / THUMB_SCALE
    h = selected_image.shape[0] / THUMB_SCALE
    tl_x = int(x_c - w/2)
    tl_y = int(y_c - h/2)
    # overlay the tile
    if update_tile:
        safe_image_broadcast(selected_image, ui_overlay, tl_x, tl_y)

    # use the same height-driven rescale as in `rescale_overview()`
    # constrain by height and aspect ratio
    (y_res, x_res, _planes) = self.overview.shape
    h_target = self.lbl_overview.height()
    scaled = cv2.resize(ui_overlay, (int(x_res * (h_target / y_res)), h_target))

    # draw the immediate selection
    thickness = ceil((y_res / h_target) * THUMB_SCALE) # get a 1-pix line after rescaling
    self.draw_rect_at_center((x_c, y_c), scaled, thickness = thickness, color = (255, 192, 255))

    # overlay the group selection preview
    if self.show_selection:
        ui_overlay = self.compute_selection_overlay()
        scaled = cv2.addWeighted(scaled, 1.0, ui_overlay, 0.5, 0.0)

    # blit to viewing portal
    height, width, planes = scaled.shape
    bytesPerLine = planes * width
    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
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
        # just select the currently selected tile
        return [self.selected_image_centroid]

    boundary = Rect(self.select_pt1, self.select_pt2)
    coords_in_range = []
    for coords in self.schema.coords_mm:
        c = Point(coords[0], coords[1])
        if boundary.intersects(c):
            coords_in_range += [coords]
    return coords_in_range

def rect_at_center(self, c):
    (x_c, y_c) = c
    w = (self.overview_actual_size[0] / self.schema.max_res[0]) * X_RES
    h = (self.overview_actual_size[1] / self.schema.max_res[1]) * Y_RES
    # define the rectangle
    x_c = (self.overview_actual_size[0] / self.schema.max_res[0]) * x_c
    y_c = (self.overview_actual_size[1] / self.schema.max_res[1]) * y_c
    tl_x = int(x_c - w/2)
    tl_y = int(y_c - h/2)
    return Rect(Point(tl_x, tl_y), Point(tl_x + int(w), tl_y + int(h)))

def draw_rect_at_center(self, c, img, thickness = 1, color = (128, 128, 128)):
    r = self.rect_at_center(c)
    cv2.rectangle(
        img,
        r.tl_int_tup(),
        r.br_int_tup(),
        color,
        thickness = thickness,
        lineType = cv2.LINE_4
    )

def compute_selection_overlay(self):
    if self.selected_image_centroid is None: # edge case of startup, nothing has been clicked yet
        return
    ui_overlay = np.zeros(self.overview_scaled.shape, self.overview_scaled.dtype)
    coords_in_range = self.get_coords_in_range()
    for coord in coords_in_range:
        (_layer, tile) = self.schema.get_tile_by_coordinate(coord)
        metadata = Schema.meta_from_tile(tile)
        (x_c, y_c) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        self.draw_rect_at_center((x_c, y_c), ui_overlay)
    return ui_overlay

def preview_selection(self):
    if not self.show_selection or self.selected_image_centroid is None:
        return
    ui_overlay = self.compute_selection_overlay()
    composite = cv2.addWeighted(self.overview_scaled, 1.0, ui_overlay, 0.5, 0.0)

    height, width, planes = self.overview_scaled.shape
    bytesPerLine = planes * width
    self.lbl_overview.setPixmap(QPixmap.fromImage(
        QImage(composite.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    ))

# ASSUME: tile is X_RES, Y_RES in resolution
def centroid_to_tile_bounding_rect_mm(self, centroid_mm):
    (x_mm, y_mm) = centroid_mm
    w_mm = (X_RES / Schema.PIX_PER_UM) / 1000
    h_mm = (Y_RES / Schema.PIX_PER_UM) / 1000

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

def on_focus_visualize(self):
    if self.status_focus_plane_button.text() == 'Visualize Focus':
        self.status_focus_plane_button.setText('Remove Focus Overlay')
        # extract and plot the raw points
        x = []
        y = []
        z = []
        layers = []
        for layer, tile in self.schema.tiles():
            meta = Schema.meta_from_fname(tile['file_name'])
            x += [meta['x'] + tile['offset'][0] / 1000]
            y += [meta['y'] + tile['offset'][1] / 1000]
            z += [(meta['z'] - meta['p'] * SECULAR_PIEZO_UM_PER_LSB / 1000.0)]
            layers += [layer]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)

        # extract the plane equation used for focus determination from the debug data
        try:
            with open(self.schema.path / Path('debug.json'), 'r') as debug_f:
                focus_debug = json.loads(debug_f.read())
                plane_poly = focus_debug['plane']
        except:
            plane_poly = None
        # plot the plane equation, if it exists
        if plane_poly is not None:
            plane_x = np.linspace(self.schema.br_frame[0], self.schema.tl_frame[0], 10)
            plane_y = np.linspace(self.schema.br_frame[1], self.schema.tl_frame[1], 10)
            PX, PY = np.meshgrid(plane_x, plane_y)
            PZ = -(plane_poly[3] + plane_poly[0] * PX + plane_poly[1] * PY) / plane_poly[2]
            ax.plot_surface(PX, PY, PZ)

        # compute the best-fit plane against the focus data
        points = np.column_stack((x, y, z))
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        U, S, Vt = svd(centered_points)
        normal_vector = Vt[-1, :]
        normal_vector /= np.linalg.norm(normal_vector)
        A, B, C = normal_vector
        D = -np.dot(normal_vector, centroid)
        PZ_prime = -(D + A * PX + B * PY) / C
        ax.plot_surface(PX, PY, PZ_prime)

        # show all of the above, plotted together
        plt.show()

        # compute the distance of the points to the best-fit plane
        dist_list = []
        for (x0, y0, z0) in points:
            dist_list += [abs(A * x0 + B * y0 + C * z0 + D) / sqrt(A**2 + B**2 + C**2)]
        dist_np = np.array(dist_list, dtype=float)
        normalized_dist = cv2.normalize(dist_np, None, 0, 1, norm_type=cv2.NORM_MINMAX)
        self.layer_dist_dict = {key: value for key, value in zip(layers, normalized_dist)}
    else:
        self.layer_dist_dict = None
        self.status_focus_plane_button.setText('Visualize Focus')

    self.redraw_overview()