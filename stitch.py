#! /usr/bin/env python3

import argparse
from pathlib import Path
import logging
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
from config import *

# TODO:
# - fix click-area selector to be more discriminate (at the moment selects way too many tiles)
# - make a "heat map" for MSE post-autostitch? (triggered by button)
# - split stitch into setup/auto phases
#    - setup aligns all the edge bits manually
#    - auto runs all stitching without check requests
# - during stitching, maybe offer mode to not render un-stitched items
#    (because the unstitched items make it hard to judge quality of the latest row due to overlaps)
# - add a tick box for MSE cleanup option (does an MSE search for improvement in each cardinal
#    direction after each pass, and gradient follows until it gets to best, then does another
#    cardinal direction until no more improvement)

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

def check_thumbnails(args):
    thumb_path = Path("raw/" + args.name + "/thumbs")
    if not thumb_path.is_dir():
        force_generate = True
        thumb_path.mkdir()
    else:
        force_generate = args.regenerate_thumbs

    # Index and load raw image data
    raw_image_path = Path("raw/" + args.name)
    files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

    # Load based on filenames, and finalize the overall area
    for file in files:
        if '_r' + str(Schema.INITIAL_R) in file.stem: # filter image revs by the initial default rev
            if force_generate or not (thumb_path / file.name).is_file:
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                thumb = cv2.resize(img, None, None, fx=THUMB_SCALE, fy=THUMB_SCALE)
                cname = "raw/" + args.name + "/thumbs/" + file.name
                logging.info(f"Thumbnail to {cname}")
                cv2.imwrite(cname, thumb, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

class MainWindow(QMainWindow):
    from mse_stitch import stitch_one_mse
    from template_stitch import stitch_one_template, stitch_auto_template_linear, restitch_one
    from blend import blend
    from zoom import update_zoom_window, on_cv_zoom, get_centered_and_scaled_image, draw_scale_bar
    from overview import redraw_overview, rescale_overview, update_selected_rect,\
        centroid_to_tile_bounding_rect_mm, snap_range, check_res_bounds,\
        pix_to_um_absolute, um_to_pix_absolute, preview_selection, get_coords_in_range,\
        compute_selection_overlay, draw_rect_at_center, rect_at_center, on_focus_visualize,\
        generate_fullres_overview, on_layer_click

    def __init__(self):
        super().__init__()

        self.setMinimumSize(UI_MIN_WIDTH, UI_MIN_HEIGHT)
        self.timer = QTimer(self)

        # Setup widget layouts
        self.status_bar = QWidget()
        self.status_bar.setMinimumWidth(150)
        self.status_bar.setMaximumWidth(350)

        status_overall_layout = QVBoxLayout()

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
        self.status_fit_metric_ui = QLabel("None")
        self.status_score_metric_ui = QLabel("None")
        self.status_ratio_metric_ui = QLabel("None")
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)
        status_fields_layout.addRow("Offset:", self.status_offset_ui)
        status_fields_layout.addRow("Focus fit:", self.status_fit_metric_ui)
        status_fields_layout.addRow("Focus fit:", self.status_score_metric_ui)
        status_fields_layout.addRow("Focus fit:", self.status_ratio_metric_ui)
        status_fields_layout.addRow("Stitch score:", self.status_score)
        status_fields_layout.addRow("Stitch error:", self.status_stitch_err)
        status_fields_layout.addRow("Rev:", self.status_rev_ui)
        status_fields_layout.addRow("Laplacian:", self.status_laplacian_ui)
        status_fields_layout.addRow("Filter:", self.status_filter_ui)
        status_fields_layout.addRow("Select Pt 1:", self.status_select_pt1_ui)
        status_fields_layout.addRow("Select Pt 2:", self.status_select_pt2_ui)
        status_overall_layout.addLayout(status_fields_layout)

        self.status_layer_select_layout = QVBoxLayout()
        status_overall_layout.addLayout(self.status_layer_select_layout)

        self.status_restitch_selection_button = QPushButton("Restitch Selection")
        self.status_restitch_selection_button.clicked.connect(self.restitch_selection)
        self.status_flag_manual_review_button = QPushButton("Flag for Manual Review")
        self.status_flag_manual_review_button.clicked.connect(self.on_flag_manual_review)
        self.status_anchor_button = QPushButton("Make Anchor")
        self.status_anchor_button.clicked.connect(self.on_anchor_button)
        self.status_autostitch_button = QPushButton("Interactive Autostitch")
        self.status_autostitch_button.clicked.connect(self.on_autostitch_button)
        self.status_save_button = QPushButton("Save Schema")
        self.status_save_button.clicked.connect(self.on_save_button)
        self.status_save_fast_button = QPushButton("Save Without Blend")
        self.status_save_fast_button.clicked.connect(self.on_fast_save_button)
        self.status_render_button = QPushButton("Render and Save")
        self.status_render_button.clicked.connect(self.on_render_button)
        self.status_redraw_button = QPushButton("Redraw")
        self.status_redraw_button.clicked.connect(self.on_redraw_button)
        self.status_preview_selection_button = QPushButton("Hide Selection")
        self.status_preview_selection_button.clicked.connect(self.on_preview_selection)
        self.status_clear_selection_button = QPushButton("Clear Selection")
        self.status_clear_selection_button.clicked.connect(self.on_clear_selection)
        self.status_cycle_rev_button = QPushButton("Cycle rev")
        self.status_cycle_rev_button.clicked.connect(self.on_cycle_rev)
        self.status_focus_plane_button = QPushButton("Visualize Focus")
        self.status_focus_plane_button.clicked.connect(self.on_focus_visualize)
        self.status_remove_tile_button = QPushButton("Remove Selected")
        self.status_remove_tile_button.clicked.connect(self.on_remove_selected)
        self.status_undo_button = QPushButton("Undo")
        self.status_undo_button.clicked.connect(self.on_undo_button)
        status_overall_layout.addWidget(self.status_redraw_button)
        status_overall_layout.addWidget(self.status_preview_selection_button)
        status_overall_layout.addWidget(self.status_clear_selection_button)
        status_overall_layout.addWidget(self.status_cycle_rev_button)
        status_overall_layout.addWidget(self.status_focus_plane_button)
        status_overall_layout.addStretch()
        status_overall_layout.addWidget(self.status_autostitch_button)
        status_overall_layout.addWidget(self.status_restitch_selection_button)
        status_overall_layout.addWidget(self.status_flag_manual_review_button)
        status_overall_layout.addWidget(self.status_remove_tile_button)
        status_overall_layout.addWidget(self.status_undo_button)
        status_overall_layout.addStretch()
        status_overall_layout.addWidget(self.status_save_button)
        status_overall_layout.addWidget(self.status_save_fast_button)
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

        self.selected_image_centroid = None
        self.zoom_scale = 1.0
        self.trackbar_created = False
        self.select_pt1 = None
        self.select_pt2 = None
        self.zoom_window_opened = False
        self.show_selection = True

        self.overview_scale = THUMB_SCALE
        self.overview = None
        self.overview_fullres = None

        self.layer_dist_dict = None
        self.layer_selected = None

    def on_autostitch_button(self):
        # undo is handled inside the restitch routine
        self.status_autostitch_button.setEnabled(False)
        self.status_restitch_selection_button.setEnabled(False)
        while self.stitch_auto_template_linear():
            logging.info("Database was modified by a remove, restarting stitch...")
            self.schema.finalize() # think we want to do this to regenerate the coordinate lists...
        self.status_autostitch_button.setEnabled(True)
        self.status_restitch_selection_button.setEnabled(True)

        # redraw the main window preview
        self.redraw_overview()
        if self.zoom_window_opened:
            self.update_zoom_window()

    def restitch_selection(self):
        # undo is handled inside the restitch routine
        restitch_list = self.get_coords_in_range()
        self.status_autostitch_button.setEnabled(False)
        self.status_restitch_selection_button.setEnabled(False)
        if restitch_list is None or len(restitch_list) == 1: # stitch just the selected tile
            (layer, tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
            logging.info(f"Restitch single tile {layer} / {tile}")
            self.schema.flag_touchup(layer)
            self.restitch_one(layer)
            self.redraw_overview()
            self.update_selected_rect(update_tile=True)
        else:
            self.stitch_auto_template_linear(stitch_list=restitch_list)
            self.redraw_overview()
        if self.zoom_window_opened:
            self.update_zoom_window()
        self.status_autostitch_button.setEnabled(True)
        self.status_restitch_selection_button.setEnabled(True)

    def on_flag_manual_review(self):
        self.schema.set_undo_checkpoint()
        restitch_list = self.get_coords_in_range()
        if restitch_list is not None:
            for restitch_item in restitch_list:
                (layer, _tile) = self.schema.get_tile_by_coordinate(restitch_item)
                self.schema.flag_restitch(layer)
        else:
            logging.warning("No region selected, doing nothing")
        if self.zoom_window_opened:
            self.update_zoom_window()
        self.redraw_overview()

    def on_remove_selected(self):
        self.schema.set_undo_checkpoint()
        (layer, _tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
        self.schema.remove_tile(layer)
        self.redraw_overview()
        if self.zoom_window_opened:
            self.update_zoom_window()

    def on_redraw_button(self):
        self.redraw_overview()
        if self.zoom_window_opened:
            self.update_zoom_window()

    def on_undo_button(self):
        self.schema.undo_to_checkpoint()
        self.redraw_overview()
        if self.zoom_window_opened:
            self.update_zoom_window()
        logging.info("Undo to last checkpoint!")

    def on_preview_selection(self):
        self.show_selection = not self.show_selection
        if not self.show_selection:
            self.status_preview_selection_button.setText("Show Selection")
        else:
            self.status_preview_selection_button.setText("Hide Selection")
        self.redraw_overview()

    def on_clear_selection(self):
        self.select_pt1 = None
        self.select_pt2 = None
        self.status_select_pt1_ui.setText("None")
        self.status_select_pt2_ui.setText("None")
        self.redraw_overview()

    def on_cycle_rev(self):
        (layer, _tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)
        self.schema.cycle_rev(layer)
        self.update_selected_rect(update_tile=True)

    def on_anchor_button(self):
        self.schema.set_undo_checkpoint()
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

    def on_fast_save_button(self):
        self.generate_fullres_overview(blend=False)
        logging.info("Saving...")
        self.schema.save_image(self.overview_fullres, modifier='fast')

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
        self.schema.save_image(self.overview_fullres)
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
            if '_r' + str(Schema.INITIAL_R) in file.stem: # filter image revs by the initial default rev
                self.schema.add_tile(file, max_x = args.max_x, max_y = args.max_y)
        self.schema.finalize()
        self.schema.set_undo_checkpoint()

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
            self.selected_image_centroid = self.schema.closest_tile_to_coord_mm((x_um, y_um))
            (layer, tile) = self.schema.get_tile_by_coordinate(self.selected_image_centroid)

            if event.button() == Qt.LeftButton:
                if tile is not None: # there can be voids due to bad images that have been removed
                    self.layer_selected = None
                    if event.modifiers() & Qt.ShiftModifier:
                        self.update_selected_rect(update_tile=True)
                    else:
                        self.update_selected_rect()

            elif event.button() == Qt.RightButton:
                self.update_zoom_window()
                self.zoom_window_opened = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_1:
            if self.select_pt1 is not None:
                self.select_pt1 = None
                self.status_select_pt1_ui.setText("None")
            elif self.selected_image_centroid is not None:
                self.select_pt1 = Point(self.selected_image_centroid[0], self.selected_image_centroid[1])
                self.status_select_pt1_ui.setText(f"{self.select_pt1.x:0.1f}, {self.select_pt1.y:0.1f}")
            self.redraw_overview() # takes time, so don't do it on every key hit including irrelevant ones
        elif event.key() == Qt.Key.Key_2:
            if self.select_pt2 is not None:
                self.select_pt2 = None
                self.status_select_pt2_ui.setText("None")
            elif self.selected_image_centroid is not None:
                self.select_pt2 = Point(self.selected_image_centroid[0], self.selected_image_centroid[1])
                self.status_select_pt2_ui.setText(f"{self.select_pt2.x:0.1f}, {self.select_pt2.y:0.1f}")
            self.redraw_overview()

def main():
    parser = argparse.ArgumentParser(description="IRIS Stitching Scripts")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=False, help="name of image directory containing raw files", default='unnamed'
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
        "--save", required=False, help="Save composite to the given file root name", type=str
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
    parser.add_argument(
        "--regenerate-thumbs", default=False, help="Force regeneration of all thumbnails"
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    Schema.set_mag(args.mag) # this must be called before we create the main window, so that the filter/laplacian values are correct by default
    Schema.set_initial_r(args.initial_r)

    if False: # run unit tests
        from prims import Rect
        Rect.test()

    app = QApplication(sys.argv)
    w = MainWindow()
    w.setGeometry(200, 200, 2000, 2400)

    w.schema = Schema(use_cache=not args.no_caching)
    w.schema.average = args.average
    w.schema.avg_qc = args.avg_qc
    if args.save[-4] == '.':
        w.schema.set_save_name(args.save[:-4])
        w.schema.set_save_type(args.save[-4:])
    else:
        w.schema.set_save_name(args.save)

    # This will read in a schema if it exists, otherwise schema will be empty
    # Schema is saved in a separate routine, overwriting the existing file at that point.
    if w.schema.read(Path("raw/" + args.name), args.max_x, args.max_y): # This was originally a try/except, but somehow this is broken in Python. Maybe some import changed the behavior of error handling??
        check_thumbnails(args) # this can change the thumbnail scale depending on the overall image size
        w.redraw_overview()
    else:
        w.new_schema(args) # needs full set of args because we need to know max extents
        w.schema.overwrite()
        check_thumbnails(args) # this can change the thumbnail scale depending on the overall image size
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
