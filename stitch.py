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
from utils import *

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
    from mse_stitch import stitch_one_mse
    from template_stitch import stitch_one_template, stitch_auto_template

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
        self.status_score = QLabel("-1")
        self.status_stitch_err = QLabel("invalid")
        self.status_rev_ui = QLabel("N/A")
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)
        status_fields_layout.addRow("Offset:", self.status_offset_ui)
        status_fields_layout.addRow("Stitch score:", self.status_score)
        status_fields_layout.addRow("Stitch error:", self.status_stitch_err)
        status_fields_layout.addRow("Rev:", self.status_rev_ui)

        status_overall_layout = QVBoxLayout()
        status_overall_layout.addLayout(status_fields_layout)
        self.status_anchor_button = QPushButton("Make Anchor")
        self.status_anchor_button.clicked.connect(self.on_anchor_button)
        self.status_save_button = QPushButton("Save Schema")
        self.status_save_button.clicked.connect(self.on_save_button)
        self.status_render_button = QPushButton("Render and Save")
        self.status_render_button.clicked.connect(self.on_render_button)
        status_overall_layout.addWidget(self.status_anchor_button)
        status_overall_layout.addWidget(self.status_save_button)
        status_overall_layout.addWidget(self.status_render_button)
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
        self.zoom_display_rect_um = None

    def on_anchor_button(self):
        cur_layer = int(self.status_layer_ui.text())
        anchor_layer = self.schema.anchor_layer_index()
        self.schema.swap_layers(cur_layer, anchor_layer)
        # set anchor layer as stitched
        self.schema.store_auto_align_result(anchor_layer, 1.0, False, set_anchor=True)
        # set previous layer as unstitched
        self.schema.store_auto_align_result(cur_layer, -1.0, False)
        self.load_schema()
        # resizeEvent will force a redraw or the region
        self.resizeEvent(None)

    def on_save_button(self):
        self.schema.overwrite()

    def on_render_button(self):
        # stash schema settings
        prev_avg = self.schema.average
        prev_avg_qc = self.schema.avg_qc
        # max out quality metrics on schema
        self.schema.average = True
        self.schema.avg_qc = True
        # reload the tiles, this time blending them to remove edges
        logging.info("Rendering final image...")
        self.load_schema(blend=True)
        logging.info("Saving...")
        if self.schema.save_name is not None:
            cv2.imwrite(self.schema.save_name, self.overview)
        # restore schema settings
        self.schema.average = prev_avg
        self.schema.avg_qc = prev_avg_qc

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

    def load_schema(self, blend=False):
        sorted_tiles = self.schema.sorted_tiles()
        canvas = np.zeros((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)
        # ones indicate regions that need to be copied
        mask = np.ones((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)

        # now read in the images
        if blend:
            corners = []
            images = []
            masks = []

        for (_index, tile) in sorted_tiles:
            logging.info(f"Loading {tile}")
            img = self.schema.get_image_from_tile(tile)
            metadata = Schema.meta_from_fname(tile['file_name'])
            (x, y) = self.um_to_pix_absolute(
                (float(metadata['x']) * 1000 + float(tile['offset'][0]),
                float(metadata['y']) * 1000 + float(tile['offset'][1]))
            )
            # move center coordinate to top left
            x -= Schema.X_RES / 2
            y -= Schema.Y_RES / 2

            if blend:
                # the mask is 255 where pixels should be copied into the final mosaic canvas. In this case, we
                # want to overlay the full image every time, so the mask is easy.
                mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
                masks += [mask]
                # The image needs to be RGB 8-bit per channel for the cv2 blending algorithm
                images += [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)]
                # the corner is the top left corner of where the image should go after alignment
                corners += [(int(x), int(y))]
            else:
                canvas, mask = safe_image_broadcast(img, canvas, x, y, mask)

        if blend:
            # this computes the full size of the resulting canvas
            dst_sz = cv2.detail.resultRoi(corners=corners, images=images)
            # set up the blender algorithm. This case uses the Burt & Adelson 1983 multiresolution
            # spline algorithm (gaussian/laplacian pyramids) with some modern refinements that
            # haven't been explicitly documented by opencv.
            blender = cv2.detail_MultiBandBlender(try_gpu=1) # GPU is wicked fast - the computation is much faster than reading in the data
            # I *think* this sets how far the blending seam should go from the edge.
            blend_strength = 5 # default from example program
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            # I read "bands" as basically how deep you want the pyramids to go
            blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            # Allocates memory for the final image
            blender.prepare(dst_sz)

            # Feed the images into the blender itself
            logging.info("Blending...")
            for (img, mask, corner) in zip(images, masks, corners):
                blender.feed(img, mask, corner)

            # Actual computational step
            canvas_rgb, _canvas_mask = blender.blend(None, None)
            # The result is a uint16 RGB image: some magic happens to prevent precision loss. This is good.
            # Re-normalize and convert to gray scale.
            canvas = cv2.normalize(canvas_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow('blended', canvas)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)

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
            'stitch_mse' : Qt.Key.Key_1,
            'stitch_pyramidal' : Qt.Key.Key_2,
            'auto_pyramidal' : Qt.Key.Key_3,
            'reset_stitch' : Qt.Key.Key_4,
        }
        qwerty_key_map = {
            'left': Qt.Key.Key_A,
            'right' : Qt.Key.Key_D,
            'up' : Qt.Key.Key_W,
            'down' : Qt.Key.Key_S,
            'rev' : Qt.Key.Key_R,
            'avg' : Qt.Key.Key_G,
            'xor' : Qt.Key.Key_X,
            'stitch_mse' : Qt.Key.Key_1,
            'stitch_pyramidal' : Qt.Key.Key_2,
            'auto_pyramidal' : Qt.Key.Key_3,
            'reset_stitch' : Qt.Key.Key_4,
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
        elif event.key() == key_map['stitch_mse']:
            self.stitch_one_mse()
            self.overview_dirty = True
        elif event.key() == key_map['stitch_pyramidal']:
            ref_img = None
            moving_img = None
            # extract the reference tile and moving tile
            for (layer, t, img) in self.schema.zoom_cache:
                meta = Schema.meta_from_tile(t)
                if layer == self.ref_layer:
                    ref_img = img
                    ref_meta = meta
                    ref_t = t
                elif layer == self.selected_layer:
                    moving_img = img
                    moving_meta = meta
                    moving_t = t

            if ref_img is None or moving_img is None:
                logging.warning("Couldn't find reference or moving image, aborting!")
            else:
                self.stitch_one_template(
                    ref_img, ref_meta, ref_t,
                    moving_img, moving_meta, moving_t, self.selected_layer
                )
                self.overview_dirty = True
        elif event.key() == key_map['auto_pyramidal']:
            self.stitch_auto_template()
            self.overview_dirty = True
        elif event.key() == key_map['reset_stitch']:
            self.schema.reset_all_align_results()

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
                logging.debug(f"{self.schema.tl_frame}, {self.schema.br_frame}")
                logging.debug(f"{point.x()}[{self.overview_actual_size[0]:.2f}], {point.y()}[{self.overview_actual_size[1]:.2f}] -> {x_um / 1000}, {y_um / 1000}")

                # now figure out which image centroid this coordinate is closest to
                # retrieve an image from disk, and cache it
                self.cached_image_centroid = self.schema.closest_tile_to_coord_mm((x_um, y_um))
                (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
                if tile is not None: # there can be voids due to bad images that have been removed
                    img = self.schema.get_image_from_tile(tile)
                    self.cached_image = img.copy()

                    if event.modifiers() & Qt.ShiftModifier and 'zoom_tl_um' in dir(self):
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
            safe_image_broadcast(img, canvas, x, y)
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
                if self.zoom_display_rect_um is not None:
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
                                self.status_offset_ui.setText(f"{t['offset'][0]:0.2f}, {t['offset'][1]:0.2f}")
                                if meta['r'] >= 0:
                                    self.status_rev_ui.setText(f"{meta['r']}")
                                else:
                                    self.status_rev_ui.setText("average")
                                self.status_score.setText(f"{t['score']:0.3f}")
                                self.status_stitch_err.setText(f"{t['auto_error']}")
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
            safe_image_broadcast(img, canvas, x, y)
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

                        ref_lap_u8 = cv2.normalize(ref_laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        mov_lap_u8 = cv2.normalize(moving_laplacian, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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
        composite = cv2.addWeighted(cropped, 1.0, ui_overlay, 0.5, 0.0)

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
    parser.add_argument(
        "--save", required=False, help="Save composite to the given filename", type=str
    )
    parser.add_argument(
        "--average", required=False, help="Average images before compositing", action="store_true", default=False
    )
    parser.add_argument(
        "--avg-qc", default=False, help="do quality checks on images before averaging (slows down loading by a lot)", action="store_true"
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
    w.schema.average = args.average
    w.schema.avg_qc = args.avg_qc
    w.schema.set_save_name(args.save)

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
