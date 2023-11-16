#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2
import sys

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer, QSignalBlocker, Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtWidgets import (QLabel, QApplication, QWidget, QDesktopWidget,
                             QCheckBox, QMessageBox, QMainWindow, QPushButton,
                             QComboBox, QSlider, QGroupBox, QGridLayout, QBoxLayout,
                             QHBoxLayout, QVBoxLayout, QMenu, QAction, QFrame,
                             QSizePolicy, QFormLayout, QLineEdit, QSpinBox)

from scipy.spatial import distance
import json

from schema import Schema
from prims import Rect, Point, ROUNDING

SCALE_BAR_WIDTH_UM = None

UI_MAX_WIDTH = 2000
UI_MAX_HEIGHT = 2000
UI_MIN_WIDTH = 1000
UI_MIN_HEIGHT = 1000

INITIAL_R = 2

TILES_VERSION = 1

# Coordinate system of OpenCV and X/Y on machine:
#
# (0,0) ----> X
# |
# v
# Y
#
# Left-Click overview: select a region to show in zoom below
# Shift Left-Click overview: raise the tile for inspection in zoom below
#
# Shift Left-Click in zoom: select the tile for modification
# Left-click in zoom: move cursors for inspection
# Right-click in zoom: set the reference tile for XOR
#
# Alignment method proceeds like this:
# 1. left click in overview minimap to select an ROI
# 2. shift click in the zoom image to set the reference (non-moving) image
# 3. right click the zoom image to set the image to move
# 4. hit 'x' to turn on XOR mode
# 5. use 'wasd' to align the image
#
# TODO:
# - Create some metric for global brightness measurement & adjacent-tile compensation
# - Create a metric to evaluate "aligned-ness"
# - Create a metric to evaluate "focused-ness"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(UI_MIN_WIDTH, UI_MIN_HEIGHT)
        self.timer = QTimer(self)

        # Setup widget layouts
        self.status_bar = QWidget()
        self.status_bar.setMinimumWidth(150)
        self.status_bar.setMaximumWidth(250)


        status_fields_layout = QFormLayout()
        self.status_centroid_ui = QLabel("0, 0")
        self.status_layer_ui = QLabel("0")
        self.status_is_anchor = QCheckBox()
        self.status_offset_ui = QLabel("0, 0")
        self.status_rev_ui = QLabel("N/A")
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)
        status_fields_layout.addRow("Offset:", self.status_offset_ui)
        status_fields_layout.addRow("Rev:", self.status_rev_ui)

        status_overall_layout = QVBoxLayout()
        status_overall_layout.addLayout(status_fields_layout)
        self.status_anchor_button = QPushButton("Make Anchor")
        self.status_anchor_button.clicked.connect(self.on_anchor_button)
        self.status_save_button = QPushButton("Save Schema")
        self.status_save_button.clicked.connect(self.on_save_button)
        status_overall_layout.addWidget(self.status_anchor_button)
        status_overall_layout.addWidget(self.status_save_button)
        self.status_bar.setLayout(status_overall_layout)

        self.lbl_overview = QLabel()
        self.lbl_zoom = QLabel()
        h_top = QHBoxLayout()
        h_top.addWidget(self.lbl_overview)
        h_top.addWidget(self.status_bar)
        v_preview = QVBoxLayout()
        v_preview.addLayout(h_top)
        v_preview.addWidget(self.lbl_zoom)
        v_widget = QWidget()
        v_widget.setLayout(v_preview)
        self.v_preview = v_preview

        self.lbl_overview.mousePressEvent = self.overview_clicked

        self.lbl_zoom.mousePressEvent = self.zoom_clicked
        self.lbl_zoom.mouseMoveEvent = self.zoom_drag

        grid_main = QGridLayout()
        grid_main.setRowStretch(0, 10) # video is on row 0, have it try to be as big as possible
        grid_main.addWidget(v_widget)
        w_main = QWidget()
        w_main.setLayout(grid_main)
        self.setCentralWidget(w_main)

        self.xor = False
        self.zoom_init = False

    def on_anchor_button(self):
        cur_layer = int(self.status_layer_ui.text())
        anchor_layer = self.schema.anchor_layer_index()
        self.schema.swap_layers(cur_layer, anchor_layer)
        self.load_schema(self.schema)
        # resizeEvent will force a redraw or the region
        self.resizeEvent(None)

    def on_save_button(self):
        self.schema.overwrite()

    def new_schema(self, args):
        # Index and load raw image data
        raw_image_path = Path("raw/" + args.name)
        self.schema.path = raw_image_path
        files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

        # Load based on filenames, and finalize the overall area
        for file in files:
            if '_r' + str(INITIAL_R) in file.stem: # filter image revs by the initial default rev
                self.schema.add_tile(file)
        self.schema.finalize(max_x = args.max_x, max_y = args.max_y)

    def load_schema(self):
        sorted_tiles = self.schema.sorted_tiles()
        canvas = np.zeros((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)

        # now read in the images
        for (_index, tile) in sorted_tiles:
            img = self.schema.get_image_from_tile(tile)
            metadata = Schema.meta_from_fname(tile['file_name'])
            (x, y) = self.um_to_pix_absolute(
                (float(metadata['x'] * 1000 + tile['offset'][0]), float(metadata['y'] * 1000 + tile['offset'][1]))
            )
            # move center coordinate to top left
            x -= Schema.X_RES / 2
            y -= Schema.Y_RES / 2
            # deal with rounding errors and integer conversions
            x = int(x)
            if x < 0: # we can end up at -1 because of fp rounding errors, that's bad. snap to 0.
                x = 0
            y = int(y) + 1
            if y < 0:
                y = 0
            # copy the image to the appointed region
            dest = canvas[y: y + Schema.Y_RES, x:x + Schema.X_RES]
            cv2.addWeighted(dest, 0, img[:dest.shape[0],:dest.shape[1]], 1, 0, dest)

        self.overview = canvas
        self.overview_dirty = False
        self.rescale_overview()

    def resizeEvent(self, event):
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

    def keyPressEvent(self, event):
        if self.zoom_init == False:
            return
        dvorak_key_map = {
            'left': Qt.Key.Key_A,
            'right' : Qt.Key.Key_E,
            'up' : Qt.Key.Key_Comma,
            'down' : Qt.Key.Key_O,
            'rev' : Qt.Key.Key_R,
            'avg' : Qt.Key.Key_G,
            'xor' : Qt.Key.Key_X,
            'stitch' : Qt.Key.Key_1,
        }
        qwerty_key_map = {
            'left': Qt.Key.Key_A,
            'right' : Qt.Key.Key_D,
            'up' : Qt.Key.Key_W,
            'down' : Qt.Key.Key_S,
            'rev' : Qt.Key.Key_R,
            'avg' : Qt.Key.Key_G,
            'xor' : Qt.Key.Key_X,
            'stitch' : Qt.Key.Key_1,
        }
        key_map = dvorak_key_map
        x = 0.0
        y = 0.0
        if event.key() == key_map['left']:
            x = -1.0 / Schema.PIX_PER_UM
        elif event.key() == key_map['right']:
            x = +1.0 / Schema.PIX_PER_UM
        elif event.key() == key_map['up']:
            y = -1.0 / Schema.PIX_PER_UM
        elif event.key() == key_map['down']:
            y = +1.0 / Schema.PIX_PER_UM
        elif event.key() == key_map['rev']:
            rev = self.schema.cycle_rev(self.selected_layer)
            self.status_rev_ui.setText(f"{rev}")
        elif event.key() == key_map['avg']:
            self.schema.set_avg(self.selected_layer)
            self.status_rev_ui.setText("average")
        elif event.key() == key_map['xor']:
            self.xor = not self.xor
        elif event.key() == key_map['stitch']:
            self.try_stitch_one()

        # have to adjust both the master DB and the cached entries
        if self.selected_layer:
            if int(self.selected_layer) != int(self.schema.anchor_layer_index()): # don't move the anchor layer!
                self.schema.adjust_offset(self.selected_layer, x, y)
                check_t = self.schema.schema['tiles'][str(self.selected_layer)]
                # print(f"after adjustment: {check_t['offset'][0]},{check_t['offset'][1]}")

        # this should update the image to reflect the tile shifts
        self.redraw_zoom_area()
        self.overview_dirty = True

    def overview_clicked(self, event):
        if isinstance(event, QMouseEvent):
            # clear state used on the zoom sub-window, as we're in a new part of the global map
            self.zoom_click_px = None
            self.zoom_selection_px = None
            self.zoom_click_um = None
            self.zoom_right_click_px = None
            self.zoom_right_click_um = None
            self.selected_layer = None
            self.ref_layer = None

            # Reload the overview image if it's dirty
            if self.overview_dirty:
                self.load_schema()

            if event.button() == Qt.LeftButton:
                self.selected_layer = None
                point = event.pos()
                ums = self.pix_to_um_absolute((point.x(), point.y()), (self.overview_actual_size[0], self.overview_actual_size[1]))
                self.roi_center_ums = ums # ROI center in ums
                (x_um, y_um) = self.roi_center_ums
                logging.debug(f"{self.schema.ll_frame}, {self.schema.ur_frame}")
                logging.debug(f"{point.x()}[{self.overview_actual_size[0]:.2f}], {point.y()}[{self.overview_actual_size[1]:.2f}] -> {x_um / 1000}, {y_um / 1000}")

                # now figure out which image centroid this coordinate is closest to
                # retrieve an image from disk, and cache it
                self.cached_image_centroid = self.schema.closest_tile_to_coord_mm((x_um, y_um))
                (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
                img = self.schema.get_image_from_tile(tile)
                self.cached_image = img.copy()

                if event.modifiers() & Qt.ShiftModifier:
                    self.update_ui(img, self.cached_image_centroid)
                    self.update_selected_rect(update_tile=True)
                else:
                    img = self.update_composite_zoom()
                    self.update_ui(img, self.cached_image_centroid)
                    self.update_selected_rect()

            elif event.button() == Qt.RightButton:
                logging.info("Right button clicked at:", event.pos())

    def update_selected_rect(self, update_tile=False):
        (x_mm, y_mm) = self.cached_image_centroid
        (x_c, y_c) = self.um_to_pix_absolute((x_mm * 1000, y_mm * 1000))
        ui_overlay = np.zeros(self.overview_scaled.shape, self.overview_scaled.dtype)

        # define the rectangle
        w = (self.overview_actual_size[0] / self.schema.max_res[0]) * self.cached_image.shape[1]
        h = (self.overview_actual_size[1] / self.schema.max_res[1]) * self.cached_image.shape[0]
        x_c = (self.overview_actual_size[0] / self.schema.max_res[0]) * x_c
        y_c = (self.overview_actual_size[1] / self.schema.max_res[1]) * y_c
        tl_x = int(x_c - w/2) + 1
        tl_y = int(y_c - h/2) + 1
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
            composite = cv2.addWeighted(self.overview_scaled, 1.0, ui_overlay, 0.5, 1.0)

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
            self.status_offset_ui.setText(f"{t['offset'][0], t['offset'][1]}")
            if md['r'] >= 0:
                self.status_rev_ui.setText(f"{md['r']}")
            else:
                self.status_rev_ui.setText("average")

    def update_composite_zoom(self):
        (x_um, y_um) = self.roi_center_ums

        # click_mm is the nominal center of the canvas image
        click_mm = (x_um / 1000, y_um / 1000)
        intersection = self.schema.get_intersecting_tiles(click_mm)

        # +2 is to allow for rounding pixels on either side of the center so we don't
        # have overflow issues as we go back and forth between fp and int data types
        canvas_xres = Schema.X_RES * 3 + 2
        canvas_yres = Schema.Y_RES * 3 + 2
        canvas = np.zeros( (canvas_yres, canvas_xres), dtype = np.uint8)
        canvas_center = (canvas_xres // 2, canvas_yres // 2)

        # now load the tiles and draw them, in order, onto the canvas
        self.schema.zoom_cache_clear()
        for (layer, t) in intersection:
            img = self.schema.get_image_from_tile(t)
            meta = Schema.meta_from_tile(t)
            center_offset_px = (
                int((float(meta['x']) * 1000 + t['offset'][0] - x_um) * Schema.PIX_PER_UM),
                int((float(meta['y']) * 1000 + t['offset'][1] - y_um) * Schema.PIX_PER_UM)
            )
            x = center_offset_px[0] - Schema.X_RES // 2 + canvas_center[0]
            y = center_offset_px[1] - Schema.Y_RES // 2 + canvas_center[1]
            canvas[
                y : y + Schema.Y_RES,
                x : x + Schema.X_RES
            ] = img
            self.schema.zoom_cache_insert(layer, t, img)

        zoom_area_px = Rect(
            Point(canvas_center[0] - Schema.X_RES // 2, canvas_center[1] - Schema.Y_RES // 2),
            Point(canvas_center[0] - Schema.X_RES // 2 + Schema.X_RES, canvas_center[1] - Schema.Y_RES // 2 + Schema.Y_RES)
        )
        self.zoom_tl_um = Point(self.roi_center_ums[0] - (Schema.X_RES / 2) / Schema.PIX_PER_UM,
                                self.roi_center_ums[1] - (Schema.Y_RES / 2) / Schema.PIX_PER_UM)
        self.zoom_tile_img = canvas[zoom_area_px.tl.y : zoom_area_px.br.y,
                                    zoom_area_px.tl.x : zoom_area_px.br.x]
        return self.zoom_tile_img

    def zoom_clicked(self, event):
        if isinstance(event, QMouseEvent):
            if event.button() == Qt.LeftButton:
                self.zoom_init = True
                # print("Left button clicked at:", event.pos())
                click_x_um = self.zoom_display_rect_um.tl.x + event.pos().x() / Schema.PIX_PER_UM
                click_y_um = self.zoom_display_rect_um.tl.y + event.pos().y() / Schema.PIX_PER_UM
                self.zoom_click_um = (click_x_um, click_y_um)
                # print(f"That is {click_x_um}um, {click_y_um}, tl: {self.zoom_display_rect_um.tl.x}, {self.zoom_display_rect_um.tl.y}")

                # For testing: reverse the computation and check that it lines up
                p_pix = Point((self.zoom_click_um[0] - self.zoom_display_rect_um.tl.x) * Schema.PIX_PER_UM,
                        (self.zoom_click_um[1] - self.zoom_display_rect_um.tl.y) * Schema.PIX_PER_UM)
                assert round(p_pix.x, ROUNDING) == round(event.pos().x(), ROUNDING)
                assert round(p_pix.y, ROUNDING) == round(event.pos().y(), ROUNDING)

                # Change the selected tile if shift is active
                self.zoom_click_px = Point(event.pos().x(), event.pos().y())
                if event.modifiers() & Qt.ShiftModifier:
                    self.zoom_selection_px = self.zoom_click_px
                    self.selected_layer = None
                    for (layer, t, img) in self.schema.zoom_cache:
                        meta = Schema.meta_from_tile(t)
                        if meta['r_um'].intersects(Point(click_x_um, click_y_um)):
                            self.selected_layer = layer
                            self.status_centroid_ui.setText(f"{meta['x']:0.2f}, {meta['y']:0.2f}")
                            self.status_layer_ui.setText(f"{layer}")
                            self.status_is_anchor.setChecked(layer == self.schema.anchor_layer_index())
                            self.status_offset_ui.setText(f"{t['offset'][0], t['offset'][1]}")
                            if meta['r'] >= 0:
                                self.status_rev_ui.setText(f"{meta['r']}")
                            else:
                                self.status_rev_ui.setText("average")
                self.redraw_zoom_area()

            # set a reference layer with the right click -- the thing we're comparing against
            elif event.button() == Qt.RightButton:
                click_x_um = self.zoom_display_rect_um.tl.x + event.pos().x() / Schema.PIX_PER_UM
                click_y_um = self.zoom_display_rect_um.tl.y + event.pos().y() / Schema.PIX_PER_UM
                self.zoom_right_click_um = (click_x_um, click_y_um)
                self.zoom_right_click_px = Point(event.pos().x(), event.pos().y())

                self.ref_layer = None
                for (layer, t, img) in self.schema.zoom_cache:
                    meta = Schema.meta_from_tile(t)
                    if meta['r_um'].intersects(Point(click_x_um, click_y_um)):
                        self.ref_layer = layer
                self.redraw_zoom_area()

    def zoom_drag(self, event):
        if event.buttons() & Qt.LeftButton:
            click_x_um = self.zoom_display_rect_um.tl.x + event.pos().x() / Schema.PIX_PER_UM
            click_y_um = self.zoom_display_rect_um.tl.y + event.pos().y() / Schema.PIX_PER_UM
            self.zoom_click_um = (click_x_um, click_y_um)
            self.zoom_click_px = (event.pos().x(), event.pos().y())
            self.update_ui(self.zoom_tile_img, self.cached_image_centroid)

    def redraw_zoom_area(self):
        # now redraw, with any new modifiers
        (x_um, y_um) = self.roi_center_ums
        canvas_xres = Schema.X_RES * 3 + 2
        canvas_yres = Schema.Y_RES * 3 + 2
        canvas = np.zeros( (canvas_yres, canvas_xres), dtype = np.uint8)
        canvas_center = (canvas_xres // 2, canvas_yres // 2)

        for (layer, t, img) in self.schema.zoom_cache:
            meta = Schema.meta_from_tile(t)
            center_offset_px = (
                int((float(meta['x']) * 1000 + t['offset'][0] - x_um) * Schema.PIX_PER_UM),
                int((float(meta['y']) * 1000 + t['offset'][1] - y_um) * Schema.PIX_PER_UM)
            )
            x = center_offset_px[0] - Schema.X_RES // 2 + canvas_center[0]
            y = center_offset_px[1] - Schema.Y_RES // 2 + canvas_center[1]

            canvas[
                y : y + Schema.Y_RES,
                x : x + Schema.X_RES
            ] = img
        if self.xor:
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
                    moving_img = img
                    moving_bounds =  Rect(
                        Point(x, y),
                        Point(x + Schema.X_RES, y + Schema.Y_RES)
                    )

            if ref_img is not None and moving_img is not None:

                if False:
                    # first re-draw moving image
                    canvas[
                        moving_bounds.tl.y : moving_bounds.br.y,
                        moving_bounds.tl.x : moving_bounds.br.x
                    ] = moving_img
                    # subtract the image
                    canvas[
                        ref_bounds.tl.y : ref_bounds.br.y,
                        ref_bounds.tl.x : ref_bounds.br.x
                    ] = ref_img - canvas[
                        ref_bounds.tl.y : ref_bounds.br.y,
                        ref_bounds.tl.x : ref_bounds.br.x
                    ]
                else:
                    roi_bounds = ref_bounds.intersection(moving_bounds)
                    if roi_bounds is not None:
                        # Cheesy way to grab the intersecting pixels: draw the two images
                        # at their respective offsets, and snapshot the pixels at each redraw
                        canvas[
                            ref_bounds.tl.y : ref_bounds.br.y,
                            ref_bounds.tl.x : ref_bounds.br.x
                        ] = ref_img
                        ref_roi = canvas[
                            roi_bounds.tl.y : roi_bounds.br.y,
                            roi_bounds.tl.x : roi_bounds.br.x
                        ].copy()
                        canvas[
                            moving_bounds.tl.y : moving_bounds.br.y,
                            moving_bounds.tl.x : moving_bounds.br.x
                        ] = moving_img
                        moving_roi = canvas[
                            roi_bounds.tl.y : roi_bounds.br.y,
                            roi_bounds.tl.x : roi_bounds.br.x
                        ].copy()

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

                        # align the medians so the difference is less?
                        # ref_median = np.median(ref_laplacian)
                        # mov_median = np.median(moving_laplacian)
                        # moving_laplacian = moving_laplacian - (mov_median - ref_median)

                        # moving_laplacian = cv2.normalize(moving_laplacian, None, alpha=np.min(ref_laplacian), beta=np.max(ref_laplacian), norm_type=cv2.NORM_MINMAX, dtype=-1)
                        corr = moving_laplacian - ref_laplacian

                        ### !---> gradient descent following stdev statistics of the difference seems...just fine?
                        print("stdev: {}, median: {}".format(np.std(corr), np.median(corr)))
                        corr_u8 = cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        def reject_outliers(data, m = 2.):
                            d = np.abs(data - np.median(data))
                            mdev = np.median(d)
                            sdev = 3 * np.std(d)
                            masked = np.ma.masked_inside(data, mdev - sdev, mdev + sdev)
                            return np.ma.filled(masked, mdev)

                        #ref_lap_med = reject_outliers(ref_laplacian)
                        #mov_lap_med = reject_outliers(moving_laplacian)

                        ref_lap_u8 = cv2.normalize(ref_laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        mov_lap_u8 = cv2.normalize(moving_laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        ## stereoBGM algorithm (opencv estimator for differences in two (stereo) images, but maybe applicable here?)
                        # window_size = 5
                        # min_disp = 32
                        # num_disp = 112-min_disp
                        # stereo = cv2.StereoSGBM_create(
                        #     minDisparity = min_disp,
                        #     numDisparities = num_disp,
                        #     blockSize = 16,
                        #     P1 = 8*3*window_size**2,
                        #     P2 = 32*33*window_size**2,
                        #     disp12MaxDiff = 1,
                        #     uniquenessRatio = 10,
                        #     speckleWindowSize = 100,
                        #     speckleRange = 32,
                        # )
                        # disp = stereo.compute(ref_lap_u8, mov_lap_u8).astype(np.float32) / 16.0
                        # cv2.imshow('left', ref_lap_u8)
                        # cv2.imshow('disparity', (disp-min_disp)/num_disp)
                        # cv2.waitKey()

                        ## optical flow motion estimation method
                        # params for ShiTomasi corner detection
                        # feature_params = dict( maxCorners = 100,
                        #                     qualityLevel = 0.01,
                        #                     minDistance = 1,
                        #                     blockSize = 15 )
                        ## Parameters for lucas kanade optical flow
                        # lk_params = dict( winSize  = (1023, 1023),
                        #                 maxLevel = 2,
                        #                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                        # p0 = cv2.goodFeaturesToTrack(ref_lap_u8, mask = None, **feature_params)
                        # p1, st, err = cv2.calcOpticalFlowPyrLK(ref_lap_u8, mov_lap_u8, p0, None, **lk_params)
                        # print(err)
                        # # Select good points
                        # if p1 is not None:
                        #     good_new = p1[st==1]
                        #     good_old = p0[st==1]
                        # # draw the tracks
                        # mask = np.zeros_like(ref_lap_u8)
                        # for i, (new, old) in enumerate(zip(good_new, good_old)):
                        #     a, b = new.ravel()
                        #     c, d = old.ravel()
                        #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (128, 128, 128), 2)
                        #     ref_lap_u8 = cv2.circle(ref_lap_u8, (int(a), int(b)), 5, (250, 250, 250), -1)
                        # img = cv2.add(ref_lap_u8, mask)
                        # cv2.imshow('frame', img)
                        # # cv2.imshow('corr', corr_u8)
                        # cv2.waitKey()

                        ## attempt at correlation
                        #corr = cv2.filter2D(ref_norm, ddepth=-1, kernel=moving_norm)
                        #corr_u8 = cv2.normalize(corr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        ## simple subtraction
                        #corr = moving_norm - ref_norm
                        ## subtraction of laplacians
                        #corr_scale = corr * 255.0
                        #corr_clip = np.clip(corr_scale, 0, 255)
                        #corr_u8 = np.uint8(corr_clip)

                        i = np.hstack((ref_lap_u8, mov_lap_u8))
                        cv2.imshow('laplacians', i)
                        canvas[
                            roi_bounds.tl.y : roi_bounds.br.y,
                            roi_bounds.tl.x : roi_bounds.br.x
                        ] = corr_u8
                    else:
                        logging.warn("No overlap found between reference and moving image!")

        zoom_area_px = Rect(
            Point(canvas_center[0] - Schema.X_RES // 2, canvas_center[1] - Schema.Y_RES // 2),
            Point(canvas_center[0] - Schema.X_RES // 2 + Schema.X_RES, canvas_center[1] - Schema.Y_RES // 2 + Schema.Y_RES)
        )

        self.zoom_tile_img = canvas[zoom_area_px.tl.y : zoom_area_px.br.y,
                                    zoom_area_px.tl.x : zoom_area_px.br.x]

        self.update_ui(self.zoom_tile_img, self.cached_image_centroid)

    def try_stitch_one(self):
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

    # zoomed_img is the opencv data of the zoomed image we're looking at
    # centroid is an (x,y) tuple that indicates the centroid of the zoomed image, specified in millimeters
    def update_ui(self, zoomed_img, centroid_mm):
        (x_um, y_um) = self.roi_center_ums
        img_shape = zoomed_img.shape
        w = self.lbl_zoom.width()
        h = self.lbl_zoom.height()

        x_off = (x_um - centroid_mm[0] * 1000) * Schema.PIX_PER_UM + img_shape[1] / 2 # remember that image.shape() is (h, w, depth)
        y_off = (y_um - centroid_mm[1] * 1000) * Schema.PIX_PER_UM + img_shape[0] / 2

        # check for rounding errors and snap to pixel within range
        x_off = self.check_res_bounds(x_off, img_shape[1])
        y_off = self.check_res_bounds(y_off, img_shape[0])

        # now compute a window of pixels to extract (snap the x_off, y_off to windows that correspond to the size of the viewing portal)
        x_range = self.snap_range(x_off, w, img_shape[1])
        y_range = self.snap_range(y_off, h, img_shape[0])

        cropped = zoomed_img[y_range[0]:y_range[1], x_range[0]:x_range[1]].copy()
        # This correlates the displayed area rectangle to actual microns
        self.zoom_display_rect_um = Rect(
            Point(self.zoom_tl_um.x + x_range[0] / Schema.PIX_PER_UM,
                  self.zoom_tl_um.y + y_range[0] / Schema.PIX_PER_UM),
            Point(self.zoom_tl_um.x + x_range[1] / Schema.PIX_PER_UM,
                  self.zoom_tl_um.y + y_range[1] / Schema.PIX_PER_UM),
        )

        # draw cross-hairs
        ui_overlay = np.zeros(cropped.shape, cropped.dtype)
        clicked_y = int(y_off - y_range[0])
        clicked_x = int(x_off - x_range[0])
        if self.zoom_click_px:
            clicked_x = self.zoom_click_px[0]
            clicked_y = self.zoom_click_px[1]
        cv2.line(ui_overlay, (0, clicked_y), (img_shape[1], clicked_y), (128, 128, 128), thickness=1)
        cv2.line(ui_overlay, (clicked_x, 0), (clicked_x, img_shape[0]), (128, 128, 128), thickness=1)

        UI_SCALE_V = 4  # denominator of UI scale
        UI_SCALE_H = 7
        # draw row intensity data
        clicked_row = cropped[clicked_y,:]
        row_range = (y_range[1] - y_range[0] - (y_range[1] - y_range[0]) // UI_SCALE_V, y_range[1] - y_range[0])
        row_excursion = row_range[1] - row_range[0] # normalization of data to the actual range
        last_point = (0, row_range[1])
        for (x, r) in enumerate(clicked_row):
            cur_point = (x, int(row_range[1] - (r / 256.0) * row_excursion))
            if x != 0:
                cv2.line(ui_overlay, last_point, cur_point, (128, 128, 128), thickness = 1)
            last_point = cur_point

        # draw col intensity data
        clicked_col = cropped[:, clicked_x]
        col_range = (0, (x_range[1] - x_range[0]) // UI_SCALE_H)
        col_excursion = col_range[1] - col_range[0]
        last_point = (0, 0)
        for (y, c) in enumerate(clicked_col):
            cur_point = (int((c / 256.0) * col_excursion), y)
            if y != 0:
                cv2.line(ui_overlay, last_point, cur_point, (128, 128, 128), thickness = 1)
            last_point = cur_point

        # draw max extents
        cv2.line(ui_overlay, (0, row_range[1] - row_excursion), (img_shape[1], row_range[1] - row_excursion), (16, 16, 16), thickness=1)
        cv2.line(ui_overlay, (col_range[1], 0), (col_range[1], img_shape[0]), (16, 16, 16), thickness=1)

        # draw scale bar
        cv2.rectangle(
            ui_overlay,
            (50, 50),
            (int(50 + SCALE_BAR_WIDTH_UM * Schema.PIX_PER_UM), 60),
            (128, 128, 128),
            thickness = 1,
            lineType = cv2.LINE_4,
        )
        cv2.putText(
            ui_overlay,
            f"{SCALE_BAR_WIDTH_UM} um",
            (50, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            bottomLeftOrigin=False
        )

        # draw click spot
        if self.zoom_selection_px:
            cv2.circle(
                ui_overlay,
                self.zoom_selection_px.as_int_tuple(),
                4,
                (128, 128, 128),
                -1
            )
        if self.zoom_right_click_px:
            cv2.circle(
                ui_overlay,
                self.zoom_right_click_px.as_int_tuple(),
                5,
                (255, 255, 255),
                2
            )

        # composite = cv2.bitwise_xor(img, ui_overlay)
        composite = cv2.addWeighted(cropped, 1.0, ui_overlay, 0.5, 1.0)

        self.lbl_zoom.setPixmap(QPixmap.fromImage(
            QImage(composite.data, w, h, w, QImage.Format.Format_Grayscale8)
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

def main():
    parser = argparse.ArgumentParser(description="IRIS Stitching Scripts")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=False, help="name of image directory containing raw files", default='338s1285-b'
    )
    parser.add_argument(
        "--max-x", required=False, help="Maximum width to tile", default=None, type=float
    )
    parser.add_argument(
        "--max-y", required=False, help="Maximum height to tile", default=None, type=float
    )
    parser.add_argument(
        "--mag", default="20", help="Specify magnification of source images (as integer)", type=int
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    global SCALE_BAR_WIDTH_UM
    if args.mag == 20:
        SCALE_BAR_WIDTH_UM = 5.0
    elif args.mag == 5:
        SCALE_BAR_WIDTH_UM = 20.0
    else:
        logging.error("Magnification parameters not defined")
        exit(0)
    Schema.set_mag(args.mag)

    if False: # run unit tests
        from prims import Rect
        Rect.test()

    app = QApplication(sys.argv)
    w = MainWindow()

    w.schema = Schema()

    # This will read in a schema if it exists, otherwise schema will be empty
    # Schema is saved in a separate routine, overwriting the existing file at that point.
    if w.schema.read(Path("raw/" + args.name)): # This was originally a try/except, but somehow this is broken in Python. Maybe some import changed the behavior of error handling??
        w.load_schema()
    else:
        w.new_schema(args) # needs full set of args because we need to know max extents
        w.schema.overwrite()
        w.load_schema()

    w.rescale_overview()
    # zoom area is initially black, nothing selected.
    ww = w.lbl_zoom.width()
    wh = w.lbl_zoom.height()
    w.lbl_zoom.setPixmap(QPixmap.fromImage(
        QImage(np.zeros((wh, ww), dtype=np.uint8), ww, wh, ww, QImage.Format.Format_Grayscale8)
    ))

    w.show()

    # this should cause all the window parameters to compute to the actual displayed size,
    # versus the mock sized used during object initialization
    w.updateGeometry()
    w.resizeEvent(None)

    # run the application. execution blocks at this line, until app quits
    app.exec_()

if __name__ == "__main__":
    main()
