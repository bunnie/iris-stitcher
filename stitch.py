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

from schema import Schema, PIX_PER_UM, X_RES, Y_RES
from prims import Rect, Point, ROUNDING

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

    def on_anchor_button(self):
        cur_layer = int(self.status_layer_ui.text())
        anchor_layer = self.schema.anchor_layer_index()
        self.schema.swap_layers(cur_layer, anchor_layer)
        self.load_schema(self.schema)
        # resizeEvent will force a redraw or the region
        self.resizeEvent(None)

    def on_save_button(self):
        self.schema.overwrite()

    # TODO: test this, it has bit-rotted
    def new_schema(self, args, schema):
        # Index and load raw image data
        raw_image_path = Path("raw/" + args.name)
        self.schema.path = raw_image_path
        files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

        # Load based on filenames, and finalize the overall area
        for file in files:
            if '_r' + str(INITIAL_R) in file.stem(): # filter image revs by the initial default rev
                self.schema.add_tile(file)
        self.schema.finalize(max_x = args.max_x, max_y = args.max_y)

        canvas = np.zeros((self.schema.y_res, self.schema.x_res), dtype=np.uint8)
        # starting point for tiling into CV image space
        cv_y = 0
        cv_x = 0
        last_coord = None
        y_was_reset = False
        # now step along each x-coordinate and fetch the y-images
        for x in self.schema.x_list:
            col_coords = []
            for c in self.schema.coords:
                if c[0] == x:
                    col_coords += [c]
            col_coords = np.array(col_coords)

            # now operate on the column list
            if last_coord is not None:
                delta_x_mm = abs(col_coords[0][0] - last_coord[0])
                delta_x_pix = int(delta_x_mm * 1000 * PIX_PER_UM)
                logging.debug(f"Stepping X by {delta_x_mm:.3f}mm -> {delta_x_pix:.3f}px")
                cv_x += delta_x_pix

                # restart the coordinate to the top
                cv_y = 0
                logging.debug(f"Resetting y coord to {cv_y}")
                y_was_reset = True
            for c in col_coords:
                img = self.schema.get_image_from_tile(
                    self.schema.get_tile_by_coordinate(
                        self.schema.closest_tile_to_coord_mm(c)
                    )
                )
                if not y_was_reset and last_coord is not None:
                    delta_y_mm = abs(c[1] - last_coord[1])
                    delta_y_pix = int(delta_y_mm * 1000 * PIX_PER_UM)
                    logging.debug(f"Stepping Y by {delta_y_mm:.3f}mm -> {delta_y_pix:.3f}px")
                    cv_y += delta_y_pix
                else:
                    y_was_reset = False

                # copy the image to the appointed region
                dest = canvas[cv_y: cv_y + Y_RES, cv_x:cv_x + X_RES]
                # Note to self: use `last_coord is None` as a marker if we should stitch or not
                # i.e., first image in the series or not.
                cv2.addWeighted(dest, 0, img, 1, 0, dest)

                last_coord = c

        cv2.imwrite('debug1.png', canvas)
        self.overview = canvas
        self.overview_dirty = False
        self.schema = schema
        self.rescale_overview()

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
            x -= X_RES / 2
            y -= Y_RES / 2
            # deal with rounding errors and integer conversions
            x = int(x)
            if x < 0: # we can end up at -1 because of fp rounding errors, that's bad. snap to 0.
                x = 0
            y = int(y) + 1
            if y < 0:
                y = 0
            # copy the image to the appointed region
            dest = canvas[y: y + Y_RES, x:x + X_RES]
            cv2.addWeighted(dest, 0, img, 1, 0, dest)

        self.overview = canvas
        self.overview_dirty = False
        self.rescale_overview()

        # Features to implement:
        #  - [done] Outline the "selected" zoom image in the global view
        #  - [done] Develop a schema (JSON file?) for storing which image goes where in the final global picture
        #    - Fields: image name, nominal centroid, fixed (bool), correction offset, "contrast" offset (might be multiple params),
        #      revision number, stuff like blend or hard crop, tile over/under...
        #    - Schema needs to work with portions of a single chip run (so we can focus on just particular areas for fixup, etc.)
        #    - Schema will eventually need to be automatically writeable
        #    - Other assumptions that could be bad, and would break things pretty awful if I got them wrong:
        #      - We don't have to do rotation correction - everything is from a single image run.
        #      - We don't have to do scale correction - again, everything from a single image run.
        #  - [done] anchor an image for tiling - this will be on the foreground in the global preview window
        #  - [done] make it so that shift-click brings an image to the foreground for anchor preview...
        #  - then on "click" what happens is the zoom region shows the "stitch" status of the immediate tile
        #
        #  - Keyboard shortcuts to rotate through image revisions in a selection
        #    - This will require tracking the "r" variable in the image array...
        #  - Keyboard shortcut to go into tiling mode:
        #    - Nearby centroids will show as candidates alpha-blended over the anchored tiles
        #    - Compute a correlation coefficient of the overlapping image
        #    - Use "wasd" keys to manually move an overlapping image initially and see how the correlation coefficient changes
        #    - Store final offset for image tile in schema
        #  - Eventually, an automated guesser for tiling, once an "anchor" image is picked
        #
        #  Old ideas:
        #  - Simple test to take the image reps and try to align them and see if quality improves
        #  - Feature extraction on images, showing feature hot spots
        #  - Some sort of algorithm that tries to evaluate "focused-ness"

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
        dvorak_key_map = {
            'left': Qt.Key.Key_A,
            'right' : Qt.Key.Key_E,
            'up' : Qt.Key.Key_Comma,
            'down' : Qt.Key.Key_O,
            'rev' : Qt.Key.Key_R,
            'avg' : Qt.Key.Key_G,
        }
        qwerty_key_map = {
            'left': Qt.Key.Key_A,
            'right' : Qt.Key.Key_D,
            'up' : Qt.Key.Key_W,
            'down' : Qt.Key.Key_S,
            'rev' : Qt.Key.Key_R,
            'avg' : Qt.Key.Key_G,
        }
        key_map = dvorak_key_map
        x = 0.0
        y = 0.0
        if event.key() == key_map['left']:
            x = -1.0 / PIX_PER_UM
        elif event.key() == key_map['right']:
            x = +1.0 / PIX_PER_UM
        elif event.key() == key_map['up']:
            y = -1.0 / PIX_PER_UM
        elif event.key() == key_map['down']:
            y = +1.0 / PIX_PER_UM
        elif event.key() == key_map['rev']:
            rev = self.schema.cycle_rev(self.selected_layer)
            self.status_rev_ui.setText(f"{rev}")
        elif event.key() == key_map['avg']:
            self.schema.set_avg(self.selected_layer)
            self.status_rev_ui.setText("average")

        # have to adjust both the master DB and the cached entries
        if self.selected_layer:
            if int(self.selected_layer) != int(self.schema.anchor_layer_index()): # don't move the anchor layer!
                self.schema.adjust_offset(self.selected_layer, x, y)

        # this should update the image to reflect the tile shifts
        self.redraw_zoom_area()
        self.overview_dirty = True

    def overview_clicked(self, event):
        if isinstance(event, QMouseEvent):
            self.zoom_click_px = None
            self.zoom_selection_px = None
            self.zoom_click_um = None
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
        canvas_xres = X_RES * 3 + 2
        canvas_yres = Y_RES * 3 + 2
        canvas = np.zeros( (canvas_yres, canvas_xres), dtype = np.uint8)
        canvas_center = (canvas_xres // 2, canvas_yres // 2)

        # now load the tiles and draw them, in order, onto the canvas
        self.schema.zoom_cache_clear()
        for (layer, t) in intersection:
            img = self.schema.get_image_from_tile(t)
            meta = Schema.meta_from_tile(t)
            center_offset_px = (
                int((float(meta['x']) * 1000 + t['offset'][0] - x_um) * PIX_PER_UM),
                int((float(meta['y']) * 1000 + t['offset'][1] - y_um) * PIX_PER_UM)
            )
            x = center_offset_px[0] - X_RES // 2 + canvas_center[0]
            y = center_offset_px[1] - Y_RES // 2 + canvas_center[1]
            canvas[
                y : y + Y_RES,
                x : x + X_RES
            ] = img
            self.schema.zoom_cache_insert(layer, t, img)

        zoom_area_px = Rect(
            Point(canvas_center[0] - X_RES // 2, canvas_center[1] - Y_RES // 2),
            Point(canvas_center[0] - X_RES // 2 + X_RES, canvas_center[1] - Y_RES // 2 + Y_RES)
        )
        self.zoom_tl_um = Point(self.roi_center_ums[0] - (X_RES / 2) / PIX_PER_UM,
                                self.roi_center_ums[1] - (Y_RES / 2) / PIX_PER_UM)
        self.zoom_tile_img = canvas[zoom_area_px.tl.y : zoom_area_px.br.y,
                                    zoom_area_px.tl.x : zoom_area_px.br.x]
        return self.zoom_tile_img

    def zoom_clicked(self, event):
        if isinstance(event, QMouseEvent):
            if event.button() == Qt.LeftButton:
                # print("Left button clicked at:", event.pos())
                click_x_um = self.zoom_display_rect_um.tl.x + event.pos().x() / PIX_PER_UM
                click_y_um = self.zoom_display_rect_um.tl.y + event.pos().y() / PIX_PER_UM
                self.zoom_click_um = (click_x_um, click_y_um)
                # print(f"That is {click_x_um}um, {click_y_um}, tl: {self.zoom_display_rect_um.tl.x}, {self.zoom_display_rect_um.tl.y}")

                # For testing: reverse the computation and check that it lines up
                p_pix = Point((self.zoom_click_um[0] - self.zoom_display_rect_um.tl.x) * PIX_PER_UM,
                        (self.zoom_click_um[1] - self.zoom_display_rect_um.tl.y) * PIX_PER_UM)
                assert round(p_pix.x, ROUNDING) == round(event.pos().x(), ROUNDING)
                assert round(p_pix.y, ROUNDING) == round(event.pos().y(), ROUNDING)

                # Change the selected tile if shift is
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

    def zoom_drag(self, event):
        if event.buttons() & Qt.LeftButton:
            click_x_um = self.zoom_display_rect_um.tl.x + event.pos().x() / PIX_PER_UM
            click_y_um = self.zoom_display_rect_um.tl.y + event.pos().y() / PIX_PER_UM
            self.zoom_click_um = (click_x_um, click_y_um)
            self.zoom_click_px = (event.pos().x(), event.pos().y())
            self.update_ui(self.zoom_tile_img, self.cached_image_centroid)

    def redraw_zoom_area(self):
        # now redraw, with any new modifiers
        (x_um, y_um) = self.roi_center_ums
        canvas_xres = X_RES * 3 + 2
        canvas_yres = Y_RES * 3 + 2
        canvas = np.zeros( (canvas_yres, canvas_xres), dtype = np.uint8)
        canvas_center = (canvas_xres // 2, canvas_yres // 2)

        for (layer, t, img) in self.schema.zoom_cache:
            meta = Schema.meta_from_tile(t)
            center_offset_px = (
                int((float(meta['x']) * 1000 + t['offset'][0] - x_um) * PIX_PER_UM),
                int((float(meta['y']) * 1000 + t['offset'][1] - y_um) * PIX_PER_UM)
            )
            x = center_offset_px[0] - X_RES // 2 + canvas_center[0]
            y = center_offset_px[1] - Y_RES // 2 + canvas_center[1]

            SEL_MARGIN = 10
            r = Rect(
                Point(x, y),
                Point(x + X_RES, y + Y_RES)
            )
            canvas[
                y : y + Y_RES,
                x : x + X_RES
            ] = img

        zoom_area_px = Rect(
            Point(canvas_center[0] - X_RES // 2, canvas_center[1] - Y_RES // 2),
            Point(canvas_center[0] - X_RES // 2 + X_RES, canvas_center[1] - Y_RES // 2 + Y_RES)
        )

        self.zoom_tile_img = canvas[zoom_area_px.tl.y : zoom_area_px.br.y,
                                    zoom_area_px.tl.x : zoom_area_px.br.x]

        self.update_ui(self.zoom_tile_img, self.cached_image_centroid)

    # zoomed_img is the opencv data of the zoomed image we're looking at
    # centroid is an (x,y) tuple that indicates the centroid of the zoomed image, specified in millimeters
    def update_ui(self, zoomed_img, centroid_mm):
        (x_um, y_um) = self.roi_center_ums
        img_shape = zoomed_img.shape
        w = self.lbl_zoom.width()
        h = self.lbl_zoom.height()

        x_off = (x_um - centroid_mm[0] * 1000) * PIX_PER_UM + img_shape[1] / 2 # remember that image.shape() is (h, w, depth)
        y_off = (y_um - centroid_mm[1] * 1000) * PIX_PER_UM + img_shape[0] / 2

        # check for rounding errors and snap to pixel within range
        x_off = self.check_res_bounds(x_off, img_shape[1])
        y_off = self.check_res_bounds(y_off, img_shape[0])

        # now compute a window of pixels to extract (snap the x_off, y_off to windows that correspond to the size of the viewing portal)
        x_range = self.snap_range(x_off, w, img_shape[1])
        y_range = self.snap_range(y_off, h, img_shape[0])

        cropped = zoomed_img[y_range[0]:y_range[1], x_range[0]:x_range[1]].copy()
        # This correlates the displayed area rectangle to actual microns
        self.zoom_display_rect_um = Rect(
            Point(self.zoom_tl_um.x + x_range[0] / PIX_PER_UM,
                  self.zoom_tl_um.y + y_range[0] / PIX_PER_UM),
            Point(self.zoom_tl_um.x + x_range[1] / PIX_PER_UM,
                  self.zoom_tl_um.y + y_range[1] / PIX_PER_UM),
        )

        # draw crosshairs
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
        SCALE_BAR_WIDTH_UM = 5.0
        cv2.rectangle(
            ui_overlay,
            (50, 50),
            (int(50 + SCALE_BAR_WIDTH_UM * PIX_PER_UM), 60),
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

        # composite = cv2.bitwise_xor(img, ui_overlay)
        composite = cv2.addWeighted(cropped, 1.0, ui_overlay, 0.5, 1.0)

        self.lbl_zoom.setPixmap(QPixmap.fromImage(
            QImage(composite.data, w, h, w, QImage.Format.Format_Grayscale8)
        ))

    # ASSUME: tile is X_RES, Y_RES in resolution
    def centroid_to_tile_bounding_rect_mm(self, centroid_mm):
       (x_mm, y_mm) = centroid_mm
       w_mm = (X_RES / PIX_PER_UM) / 1000
       h_mm = (Y_RES / PIX_PER_UM) / 1000

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
            x * (self.schema.max_res[0] / res_x) / PIX_PER_UM + self.schema.x_min_mm * 1000,
            y * (self.schema.max_res[1] / res_y) / PIX_PER_UM + self.schema.y_min_mm * 1000
        )
    def um_to_pix_absolute(self, um):
        (x_um, y_um) = um
        return (
            int((x_um - self.schema.x_min_mm * 1000) * PIX_PER_UM),
            int((y_um - self.schema.y_min_mm * 1000) * PIX_PER_UM)
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
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    if True: # run unit tests
        from prims import Rect
        Rect.test()

    app = QApplication(sys.argv)
    w = MainWindow()

    schema = Schema()

    # This will read in a schema if it exists, otherwise schema will be empty
    # Schema is saved in a separate routine, overwriting the existing file at that point.
    try:
        schema.read(Path("raw/" + args.name))
        w.schema = schema
        w.load_schema()
    except FileNotFoundError:
        w.new_schema(args, schema) # needs full set of args because we need to know max extents
        schema.overwrite()

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
