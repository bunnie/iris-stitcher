#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
import cv2
import sys

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (QLabel, QApplication, QWidget, QDesktopWidget,
                             QCheckBox, QMessageBox, QMainWindow, QPushButton,
                             QComboBox, QSlider, QGroupBox, QGridLayout, QBoxLayout,
                             QHBoxLayout, QVBoxLayout, QMenu, QAction, QFrame,
                             QSizePolicy, QFormLayout, QLineEdit, QSpinBox)

from schema import Schema
from prims import Rect, Point, ROUNDING
from utils import *

# TODO:
#  - make a button for saving without blending
#  - make a button for removing a tile
#  - make a undo-db.json file for saving removed tiles, and tiles positions before re-stitching (for undo stack)
#  - make it so that clicking on the same area of overview toggles through images

SCALE_BAR_WIDTH_UM = None
BLEND_FINAL = True # toggles final rendering blending on or off

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
    from template_stitch import stitch_one_template, stitch_auto_template_linear, restitch_one
    from blend import blend
    from zoom import update_zoom_window, on_cv_zoom, get_centered_and_scaled_image
    from overview import redraw_overview, rescale_overview, update_selected_rect,\
        centroid_to_tile_bounding_rect_mm, snap_range, check_res_bounds,\
        pix_to_um_absolute, um_to_pix_absolute, preview_selection, get_coords_in_range

    def __init__(self):
        super().__init__()

        self.setMinimumSize(UI_MIN_WIDTH, UI_MIN_HEIGHT)
        self.timer = QTimer(self)

        # Setup widget layouts
        self.status_bar = QWidget()
        self.status_bar.setMinimumWidth(150)
        self.status_bar.setMaximumWidth(350)

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
        self.status_select_pt1_ui = QLabel("None")
        self.status_select_pt2_ui = QLabel("None")
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)
        status_fields_layout.addRow("Offset:", self.status_offset_ui)
        status_fields_layout.addRow("Stitch score:", self.status_score)
        status_fields_layout.addRow("Stitch error:", self.status_stitch_err)
        status_fields_layout.addRow("Rev:", self.status_rev_ui)
        status_fields_layout.addRow("Laplacian:", self.status_laplacian_ui)
        status_fields_layout.addRow("Filter:", self.status_filter_ui)
        status_fields_layout.addRow("Select Pt 1:", self.status_select_pt1_ui)
        status_fields_layout.addRow("Select Pt 2:", self.status_select_pt2_ui)

        status_overall_layout = QVBoxLayout()
        status_overall_layout.addLayout(status_fields_layout)
        self.status_flag_restitch_button = QPushButton("Restitch Selected")
        self.status_flag_restitch_button.clicked.connect(self.on_flag_restitch_button)
        self.status_flag_restitch_after_button = QPushButton("Reset Stitch In Selection")
        self.status_flag_restitch_after_button.clicked.connect(self.on_flag_restitch_after_button)
        self.status_anchor_button = QPushButton("Make Anchor")
        self.status_anchor_button.clicked.connect(self.on_anchor_button)
        self.status_autostitch_button = QPushButton("Interactive Autostitch")
        self.status_autostitch_button.clicked.connect(self.on_autostitch_button)
        self.status_save_button = QPushButton("Save Schema")
        self.status_save_button.clicked.connect(self.on_save_button)
        self.status_render_button = QPushButton("Render and Save")
        self.status_render_button.clicked.connect(self.on_render_button)
        self.status_redraw_button = QPushButton("Redraw")
        self.status_redraw_button.clicked.connect(self.on_redraw_button)
        self.status_preview_selection_button = QPushButton("Preview Selection")
        self.status_preview_selection_button.clicked.connect(self.on_preview_selection)
        self.status_clear_selection_button = QPushButton("Clear Selection")
        self.status_clear_selection_button.clicked.connect(self.on_clear_selection)
        status_overall_layout.addWidget(self.status_redraw_button)
        status_overall_layout.addWidget(self.status_preview_selection_button)
        status_overall_layout.addWidget(self.status_clear_selection_button)
        status_overall_layout.addStretch()
        status_overall_layout.addWidget(self.status_flag_restitch_button)
        status_overall_layout.addWidget(self.status_autostitch_button)
        # I think we don't need this feature anymore with the new stitching algorithm, always starts at top left
        # status_overall_layout.addWidget(self.status_anchor_button)
        status_overall_layout.addWidget(self.status_flag_restitch_after_button)
        status_overall_layout.addStretch()
        status_overall_layout.addWidget(self.status_save_button)
        status_overall_layout.addWidget(self.status_render_button)
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

        self.cached_image_centroid = None
        self.zoom_scale = 1.0
        self.trackbar_created = False
        self.select_pt1 = None
        self.select_pt2 = None

    def on_autostitch_button(self):
        self.status_autostitch_button.setEnabled(False)
        self.status_flag_restitch_button.setEnabled(False)
        while self.stitch_auto_template_linear():
            logging.info("Database was modified by a remove, restarting stitch...")
            self.schema.finalize() # think we want to do this to regenerate the coordinate lists...
        self.status_autostitch_button.setEnabled(True)
        self.status_flag_restitch_button.setEnabled(True)

        # redraw the main window preview
        self.redraw_overview()

    def on_flag_restitch_button(self):
        restitch_list = self.get_coords_in_range()
        if restitch_list is None: # stitch just the selected tile
            self.status_autostitch_button.setEnabled(False)
            self.status_flag_restitch_button.setEnabled(False)
            (layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
            tile['auto_error'] = 'true'
            self.restitch_one(layer)
            self.redraw_overview()
            self.update_selected_rect(update_tile=True)
            self.status_autostitch_button.setEnabled(True)
            self.status_flag_restitch_button.setEnabled(True)
        else:
            self.stitch_auto_template_linear(stitch_list=restitch_list)

    def on_flag_restitch_after_button(self):
        restitch_list = self.get_coords_in_range()
        if restitch_list is not None:
            for restitch_item in restitch_list:
                (_layer, tile) = self.schema.get_tile_by_coordinate(restitch_item)
                metadata = Schema.meta_from_tile(tile)
                tile['auto_error'] = 'invalid'
        else:
            logging.warning("No region selected, doing nothing")

        if False: # this resets everything after the current point...
            (_layer, tile) = self.schema.get_tile_by_coordinate(self.cached_image_centroid)
            metadata = Schema.meta_from_tile(tile)
            # we want to flag every tile below and to the right of this as requiring a restitch
            # but *not including* the flagged tile (which is probably 'invalid' and getting auto-review)
            base_x_mm = metadata['x']
            base_y_mm = metadata['y']
            for (_layer, t) in self.schema.tiles():
                m = Schema.meta_from_tile(t)
                if m['x'] > base_x_mm: # columns to the right
                    t['auto_error'] = 'invalid'
                else:
                    if m['x'] == base_x_mm and m['y'] > base_y_mm:
                        t['auto_error'] = 'invalid'
                    else:
                        pass

    def on_redraw_button(self):
        self.redraw_overview()

    def on_preview_selection(self):
        self.preview_selection()

    def on_clear_selection(self):
        self.select_pt1 = None
        self.select_pt2 = None
        self.status_select_pt1_ui.setText("None")
        self.status_select_pt2_ui.setText("None")

    def on_anchor_button(self):
        cur_layer = int(self.status_layer_ui.text())
        anchor_layer = self.schema.anchor_layer_index()
        if cur_layer != anchor_layer:
            self.schema.swap_layers(cur_layer, anchor_layer)
            # set anchor layer as stitched
            self.schema.store_auto_align_result(anchor_layer, 1.0, False, set_anchor=True)
            # set previous layer as unstitched
            self.schema.store_auto_align_result(cur_layer, -1.0, False)
            self.redraw_overview()

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
        if BLEND_FINAL:
            self.blend()
        else:
            self.redraw_overview(blend=False)
        logging.info("Saving...")
        if self.schema.save_name is not None:
            cv2.imwrite(self.schema.save_name, self.overview)
        # restore schema settings
        self.schema.average = prev_avg
        self.schema.avg_qc = prev_avg_qc

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

    def resizeEvent(self, event):
        self.rescale_overview()

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

            if event.button() == Qt.LeftButton:
                if tile is not None: # there can be voids due to bad images that have been removed
                    img = self.schema.get_image_from_layer(layer)
                    self.cached_image = img.copy()

                    if event.modifiers() & Qt.ShiftModifier:
                        self.update_selected_rect(update_tile=True)
                    else:
                        self.update_selected_rect()

            elif event.button() == Qt.RightButton:
                self.update_zoom_window()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_1:
            if self.cached_image_centroid is not None:
                self.select_pt1 = Point(self.cached_image_centroid[0], self.cached_image_centroid[1])
                self.status_select_pt1_ui.setText(f"{self.select_pt1.x:0.1f}, {self.select_pt1.y:0.1f}")
        elif event.key() == Qt.Key.Key_2:
            if self.cached_image_centroid is not None:
                self.select_pt2 = Point(self.cached_image_centroid[0], self.cached_image_centroid[1])
                self.status_select_pt2_ui.setText(f"{self.select_pt2.x:0.1f}, {self.select_pt2.y:0.1f}")

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
        w.redraw_overview()
    else:
        w.new_schema(args) # needs full set of args because we need to know max extents
        w.schema.overwrite()
        w.redraw_overview()

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
