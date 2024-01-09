#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
import numpy as np
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

UI_MIN_WIDTH = 1000
UI_MIN_HEIGHT = 1000

INITIAL_R = 1

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

class MainWindow(QMainWindow):
    from mse_stitch import stitch_one_mse
    from template_stitch import stitch_one_template, stitch_auto_template_linear
    from blend import blend

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
        self.status_laplacian_ui = QSpinBox()
        self.status_laplacian_ui.setRange(1, 31)
        self.status_laplacian_ui.setSingleStep(2)
        self.status_laplacian_ui.setValue(Schema.LAPLACIAN_WINDOW)
        self.status_laplacian_ui.valueChanged.connect(self.on_laplacian_changed)
        self.status_filter_ui = QSpinBox()
        self.status_filter_ui.setRange(-1, 31)
        self.status_filter_ui.setSingleStep(2)
        self.status_filter_ui.setValue(Schema.FILTER_WINDOW)
        self.status_filter_ui.valueChanged.connect(self.on_filter_changed)
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)
        status_fields_layout.addRow("Offset:", self.status_offset_ui)
        status_fields_layout.addRow("Stitch score:", self.status_score)
        status_fields_layout.addRow("Stitch error:", self.status_stitch_err)
        status_fields_layout.addRow("Rev:", self.status_rev_ui)
        status_fields_layout.addRow("Laplacian:", self.status_laplacian_ui)
        status_fields_layout.addRow("Filter:", self.status_filter_ui)

        status_overall_layout = QVBoxLayout()
        status_overall_layout.addLayout(status_fields_layout)
        self.status_flag_restitch_button = QPushButton("Flag for Restitch")
        self.status_flag_restitch_button.clicked.connect(self.on_flag_restitch_button)
        self.status_anchor_button = QPushButton("Make Anchor")
        self.status_anchor_button.clicked.connect(self.on_anchor_button)
        self.status_autostitch_button = QPushButton("Interactive Autostitch")
        self.status_autostitch_button.clicked.connect(self.on_autostitch_button)
        self.status_save_button = QPushButton("Save Schema")
        self.status_save_button.clicked.connect(self.on_save_button)
        self.status_render_button = QPushButton("Render and Save")
        self.status_render_button.clicked.connect(self.on_render_button)
        self.status_redraw_button = QPushButton("Redraw Composite")
        self.status_redraw_button.clicked.connect(self.on_redraw_button)
        status_overall_layout.addWidget(self.status_flag_restitch_button)
        status_overall_layout.addWidget(self.status_anchor_button)
        status_overall_layout.addWidget(self.status_autostitch_button)
        status_overall_layout.addWidget(self.status_save_button)
        status_overall_layout.addWidget(self.status_render_button)
        status_overall_layout.addWidget(self.status_redraw_button)
        self.status_bar.setLayout(status_overall_layout)

        self.lbl_overview = QLabel()
        h_top = QHBoxLayout()
        h_top.addWidget(self.lbl_overview)
        h_top.addWidget(self.status_bar)
        v_preview = QVBoxLayout()
        v_preview.addLayout(h_top)
        v_widget = QWidget()
        v_widget.setLayout(v_preview)
        self.v_preview = v_preview

        self.lbl_overview.mousePressEvent = self.overview_clicked

        grid_main = QGridLayout()
        grid_main.setRowStretch(0, 10) # video is on row 0, have it try to be as big as possible
        grid_main.addWidget(v_widget)
        w_main = QWidget()
        w_main.setLayout(grid_main)
        self.setCentralWidget(w_main)

    def on_flag_restitch_button(self):
        (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
        tile['auto_error'] = 'true'
        self.update_selected_rect(update_tile=True)

    def on_redraw_button(self):
        self.load_schema()

    def on_anchor_button(self):
        cur_layer = int(self.status_layer_ui.text())
        anchor_layer = self.schema.anchor_layer_index()
        if cur_layer != anchor_layer:
            self.schema.swap_layers(cur_layer, anchor_layer)
            # set anchor layer as stitched
            self.schema.store_auto_align_result(anchor_layer, 1.0, False, set_anchor=True)
            # set previous layer as unstitched
            self.schema.store_auto_align_result(cur_layer, -1.0, False)
            self.load_schema()

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
        self.blend()
        logging.info("Saving...")
        if self.schema.save_name is not None:
            cv2.imwrite(self.schema.save_name, self.overview)
        # restore schema settings
        self.schema.average = prev_avg
        self.schema.avg_qc = prev_avg_qc

    def on_autostitch_button(self):
        while self.stitch_auto_template_linear():
            logging.info("Database was modified by a remove, restarting stitch...")
        self.oveview_dirty = True

        # redraw the main window preview
        self.load_schema()

    def on_laplacian_changed(self, value):
        Schema.set_laplacian(value)
    def on_filter_changed(self, value):
        Schema.set_filter(value)

    def new_schema(self, args):
        # Index and load raw image data
        raw_image_path = Path("raw/" + args.name)
        self.schema.path = raw_image_path
        files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

        # Load based on filenames, and finalize the overall area
        for file in files:
            if '_r' + str(INITIAL_R) in file.stem: # filter image revs by the initial default rev
                self.schema.add_tile(file, max_x = args.max_x, max_y = args.max_y)
        self.schema.finalize(max_x = args.max_x, max_y = args.max_y)

    def load_schema(self):
        sorted_tiles = self.schema.sorted_tiles()
        canvas = np.zeros((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)
        # ones indicate regions that need to be copied
        mask = np.ones((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)

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
                (layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
                if tile is not None: # there can be voids due to bad images that have been removed
                    img = self.schema.get_image_from_layer(layer)
                    self.cached_image = img.copy()

                    if event.modifiers() & Qt.ShiftModifier:
                        self.update_selected_rect(update_tile=True)
                    else:
                        self.update_selected_rect()

            elif event.button() == Qt.RightButton:
                logging.info("Right button clicked at:", event.pos())

    def update_selected_rect(self, update_tile=False):
        (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
        metadata = Schema.meta_from_tile(tile)
        x_mm = metadata['x'] + tile['offset'][0] / 1000
        y_mm = metadata['y'] + tile['offset'][1] / 1000

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
        "--mag", default="10", help="Specify magnification of source images (as integer)", type=int, choices=[5, 10, 20]
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
    parser.add_argument(
        "--initial-r", default=1, help="Initial photo rep # for rough stitching", type=int
    )
    parser.add_argument(
        "--no-caching", default=False, help="Use this argument if you're running out of RAM. Slows down composite redraws significantly.", action="store_true"
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
    elif args.mag == 10:
        SCALE_BAR_WIDTH_UM = 10.0
    else:
        logging.error("Magnification parameters not defined")
        exit(0)
    global INITIAL_R
    INITIAL_R = args.initial_r
    Schema.set_mag(args.mag) # this must be called before we create the main window, so that the filter/laplacian values are correct by default

    if False: # run unit tests
        from prims import Rect
        Rect.test()

    app = QApplication(sys.argv)
    w = MainWindow()
    w.setGeometry(200, 200, 2000, 2400)

    w.schema = Schema(use_cache=not args.no_caching)
    w.schema.average = args.average
    w.schema.avg_qc = args.avg_qc
    w.schema.set_save_name(args.save)

    # This will read in a schema if it exists, otherwise schema will be empty
    # Schema is saved in a separate routine, overwriting the existing file at that point.
    if w.schema.read(Path("raw/" + args.name), args.max_x, args.max_y): # This was originally a try/except, but somehow this is broken in Python. Maybe some import changed the behavior of error handling??
        w.load_schema()
    else:
        w.new_schema(args) # needs full set of args because we need to know max extents
        w.schema.overwrite()
        w.load_schema()

    w.rescale_overview()

    w.show()

    # this should cause all the window parameters to compute to the actual displayed size,
    # versus the mock sized used during object initialization
    w.updateGeometry()
    w.resizeEvent(None)

    # run the application. execution blocks at this line, until app quits
    app.exec_()

if __name__ == "__main__":
    main()
