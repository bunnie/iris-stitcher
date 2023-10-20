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

# derived from reference image "full-H"
# NOTE: this may change with improvements in the microscope hardware.
# be sure to re-calibrate after adjustments to the hardware.
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

UI_MAX_WIDTH = 2000
UI_MAX_HEIGHT = 2000
UI_MIN_WIDTH = 1000
UI_MIN_HEIGHT = 1000

INITIAL_R = 2

TILES_VERSION = 1

class MainWindow(QMainWindow):
    def __init__(self, chip_name):
        super().__init__()
        self.chip_name = chip_name

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
        status_fields_layout.addRow("Centroid:", self.status_centroid_ui)
        status_fields_layout.addRow("Layer:", self.status_layer_ui)
        status_fields_layout.addRow("Is anchor:", self.status_is_anchor)

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
        self.lbl_overview.mouseMoveEvent = self.overview_drag

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
        self.schema.overwrite(Path("raw/" + self.chip_name + "/db.json"))

    def new_schema(self, args, schema):
        # Index and load raw image data
        raw_image_path = Path("raw/" + args.name)
        files = [file for file in raw_image_path.glob('*.png') if file.is_file()]
        self.files = files

        # Coordinate system of OpenCV and X/Y on machine:
        #
        # (0,0) ----> X
        # |
        # v
        # Y

        centroids = []
        for file in files:
            elems = file.stem.split('_')
            x = None
            y = None
            for e in elems:
                if 'x' in e:
                    x = float(e[1:])
                if 'y' in e:
                    y = float(e[1:])
            if (x is not None and y is None) or (y is not None and x is None):
                logging.error(f"only one coordinate found in {file.stem}")
            else:
                if x is not None and y is not None:
                    centroids += [[x, y]]

        coords = np.unique(centroids, axis=0)

        # Find the "lower left" corner. This is done by computing the euclidian distance
        # from all the points to a point at "very lower left", i.e. -100, -100
        dists = []
        for p in coords:
            dists += [np.linalg.norm(p - [-100, -100])]
        ll_centroid = coords[dists.index(min(dists))]
        ur_centroid = coords[dists.index(max(dists))]
        logging.info(f"Raw data: Lower-left coordinate: {ll_centroid}; upper-right coordinate: {ur_centroid}")

        if args.max_x:
            coords = [c for c in coords if c[0] <= ll_centroid[0] + args.max_x]
        if args.max_y:
            coords = [c for c in coords if c[1] <= ll_centroid[1] + args.max_y]

        if args.max_x is not None or args.max_y is not None:
            coords = np.array(coords)
            # redo the ll/ur computations
            dists = []
            for p in coords:
                dists += [np.linalg.norm(p - [-100, -100])]
            ll_centroid = coords[dists.index(min(dists))]
            ur_centroid = coords[dists.index(max(dists))]
            logging.info(f"Reduced data: Lower-left coordinate: {ll_centroid}; upper-right coordinate: {ur_centroid}")

        # note that ur, ll are the coordinates of the center of the images forming the tiles. This means
        # the actual region shown is larger, because the images extend out from the center of the images.

        # Determine total area of imaging centroid
        x_mm_centroid = ur_centroid[0] - ll_centroid[0]
        y_mm_centroid = ur_centroid[1] - ll_centroid[1]
        # Determine absolute imaging area in pixels based on pixels/mm and image size
        # X_RES, Y_RES added because we have a total of one frame size surrounding the centroid
        x_res = int(math.ceil(x_mm_centroid * 1000 * PIX_PER_UM + X_RES))
        y_res = int(math.ceil(y_mm_centroid * 1000 * PIX_PER_UM + Y_RES))
        logging.info(f"Final image resolution is {x_res}x{y_res}")
        # resolution of total area
        self.max_res = (x_res, y_res)

        self.ll_frame = [ll_centroid[0] - (X_RES / (2 * PIX_PER_UM)) / 1000, ll_centroid[1] - (Y_RES / (2 * PIX_PER_UM)) / 1000]
        self.ur_frame = [ur_centroid[0] + (X_RES / (2 * PIX_PER_UM)) / 1000, ur_centroid[1] + (Y_RES / (2 * PIX_PER_UM)) / 1000]

        canvas = np.zeros((y_res, x_res), dtype=np.uint8)

        # Build the explorer window
        if y_res > x_res:
            height = UI_MAX_HEIGHT
            width = (UI_MAX_HEIGHT / y_res) * x_res
        else:
            width = UI_MAX_WIDTH
            height = (UI_MAX_WIDTH / x_res) * y_res

        # create a list of x-coordinates
        self.x_list = x_list = np.unique(np.rot90(coords)[1])
        self.y_list = np.unique(np.rot90(coords)[0])
        self.coords = [tuple(coord) for coord in coords] # list of unique coordinates of image centroids, as tuples

        self.x_min_mm = self.ll_frame[0]
        self.y_min_mm = self.ll_frame[1]

        # starting point for tiling into CV image space
        cv_y = 0
        cv_x = 0
        last_coord = None
        y_was_reset = False
        # now step along each x-coordinate and fetch the y-images
        for x in x_list:
            col_coords = []
            for c in coords:
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
                (img, fname) = self.get_image(c, r=INITIAL_R)
                schema.add_tile(fname)
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
        self.schema = schema

    def finalize_ui_load(self):
        w = self.lbl_overview.width()
        h = self.lbl_overview.height()
        # constrain by height and aspect ratio
        scaled = cv2.resize(self.overview, (int(self.max_res[0] * (h / self.max_res[1])), h))
        height, width = scaled.shape
        bytesPerLine = 1 * width
        self.lbl_overview.setPixmap(QPixmap.fromImage(
            QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
        ))
        self.overview_actual_size = (width, height)

        (zoom_initial, _fname) = self.get_image(self.coords[0], r=INITIAL_R)
        scaled_zoom_initial = cv2.resize(zoom_initial, (int(self.max_res[0] * (h / self.max_res[1])), h))
        self.lbl_zoom.setPixmap(QPixmap.fromImage(
            QImage(scaled_zoom_initial.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
        ))
        # stash a copy for restoring after doing UX overlays
        self.overview_scaled = scaled_zoom_initial.copy()

    def load_schema(self, schema):
        sorted_tiles = schema.sorted_tiles()
        self.coords = []
        # first, extract the full extent of the data we plan to read in so we can allocate memory accordingly
        for (index, tile) in sorted_tiles:
            metadata = Schema.parse_meta(tile['file_name'])
            self.coords += [(metadata['x'], metadata['y'])]

        self.x_list = np.unique(np.rot90(self.coords)[1])
        self.y_list = np.unique(np.rot90(self.coords)[0])
        ll_centroid = (min(self.x_list), min(self.y_list))
        ur_centroid = (max(self.x_list), max(self.y_list))
        logging.info(f"Found: Lower-left coordinate: {ll_centroid}; upper-right coordinate: {ur_centroid}")

        self.ll_frame = [ll_centroid[0] - (X_RES / (2 * PIX_PER_UM)) / 1000, ll_centroid[1] - (Y_RES / (2 * PIX_PER_UM)) / 1000]
        self.ur_frame = [ur_centroid[0] + (X_RES / (2 * PIX_PER_UM)) / 1000, ur_centroid[1] + (Y_RES / (2 * PIX_PER_UM)) / 1000]
        self.x_min_mm = self.ll_frame[0]
        self.y_min_mm = self.ll_frame[1]

        x_mm_centroid = ur_centroid[0] - ll_centroid[0]
        y_mm_centroid = ur_centroid[1] - ll_centroid[1]
        x_res = int(math.ceil(x_mm_centroid * 1000 * PIX_PER_UM + X_RES))
        y_res = int(math.ceil(y_mm_centroid * 1000 * PIX_PER_UM + Y_RES))
        logging.info(f"Final image resolution is {x_res}x{y_res}")
        self.max_res = (x_res, y_res)

        canvas = np.zeros((y_res, x_res), dtype=np.uint8)

        # now read in the images
        self.files = [] # maintain the file list cache for the dynamic image loader to use; this might get refactored soon-ish
        for (index, tile) in sorted_tiles:
            fname = "raw/" + self.chip_name + "/" + tile['file_name'] + '.png'
            self.files += [Path(fname)]
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            assert tile['norm_method'] == 'MINMAX', "unsupported normalization method" # we only support one for now
            img = cv2.normalize(img, None, alpha=float(tile['norm_a']), beta=float(tile['norm_b']), norm_type=cv2.NORM_MINMAX)
            metadata = Schema.parse_meta(tile['file_name'])
            (x, y) = self.um_to_pix_absolute(
                (float(metadata['x'] * 1000), float(metadata['y'] * 1000)),
                (x_res, y_res)
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
        self.schema = schema

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
        w = self.lbl_overview.width()
        h = self.lbl_overview.height()
        (x_res, y_res) = self.max_res
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
            if event.button() == Qt.LeftButton:
                # print("Left button clicked at:", event.pos())
                point = event.pos()
                ums = self.pix_to_um_absolute((point.x(), point.y()), (self.overview_actual_size[0], self.overview_actual_size[1]))
                (x_um, y_um) = ums
                logging.debug(f"{self.ll_frame}, {self.ur_frame}")
                logging.debug(f"{point.x()}[{self.overview_actual_size[0]:.2f}], {point.y()}[{self.overview_actual_size[1]:.2f}] -> {x_um / 1000}, {y_um / 1000}")

                # now figure out which image centroid this coordinate is closest to
                distances = distance.cdist(self.coords, [(x_um / 1000, y_um / 1000)])
                closest = self.coords[np.argmin(distances)]

                # retrieve an image from disk, and cache it
                (img, _fname) = self.get_image((closest[0], closest[1]), 2)
                self.cached_image = img.copy()
                self.cached_image_centroid = closest

                if event.modifiers() & Qt.ShiftModifier:
                    self.update_ui(img, closest, ums)
                    self.update_selected_rect(update_tile=True)
                else:
                    self.update_stitching_ui(img, closest, ums)
                    self.update_selected_rect()

            elif event.button() == Qt.RightButton:
                logging.info("Right button clicked at:", event.pos())

    def update_selected_rect(self, update_tile=False):
        (x_mm, y_mm) = self.cached_image_centroid
        (x_c, y_c) = self.um_to_pix_absolute((x_mm * 1000, y_mm * 1000), (self.overview_actual_size[0], self.overview_actual_size[1]))
        ui_overlay = np.zeros(self.overview_scaled.shape, self.overview_scaled.dtype)

        # define the rectangle
        w = (self.overview_actual_size[0] / self.max_res[0]) * self.cached_image.shape[1]
        h = (self.overview_actual_size[1] / self.max_res[1]) * self.cached_image.shape[0]
        x_c = (self.overview_actual_size[0] / self.max_res[0]) * x_c
        y_c = (self.overview_actual_size[1] / self.max_res[1]) * y_c
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
            md = Schema.parse_meta(t['file_name'])
            self.status_centroid_ui.setText(f"{md['x']:0.2f}, {md['y']:0.2f}")
            self.status_layer_ui.setText(f"{layer}")
            self.status_is_anchor.setChecked(layer == self.schema.anchor_layer_index())

    def overview_drag(self, event):
        if event.buttons() & Qt.LeftButton:
            point = event.pos()
            # this operates on the cached image, making drag go a bit faster
            ums = self.pix_to_um_absolute((point.x(), point.y()), (self.overview_actual_size[0], self.overview_actual_size[1]))
            self.update_ui(self.cached_image, self.cached_image_centroid, ums)

    def get_image(self, coord, r):
        img_re = re.compile('x([0-9.\-]*)_y([0-9.\-]*)_.*_r([\d*])')
        for file in self.files:
            match = img_re.match(file.stem).groups()
            if len(match) == 3:
                if coord[0] == float(match[0]) and coord[1] == float(match[1]) and r == int(match[2]):
                    logging.info(f"loading {str(file)}")
                    img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    return (img, file)
        logging.error(f"Requested file was not found at {coord}, r={r}")
        return None

    def update_stitching_ui(self, zoomed_img, centroid_mm, click_um):
        # TODO
        # make a new version of self.update_ui that displays not just the img
        # but instead shows the composite result in the zoom window...

        pass

    # zoomed_img is the opencv data of the zoomed image we're looking at
    # centroid is an (x,y) tuple that indicates the centroid of the zoomed image, specified in millimeters
    # click_um is an (x,y) tuple the location of the click on the global image in microns
    def update_ui(self, zoomed_img, centroid_mm, click_um):
        (x_um, y_um) = click_um
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

        # draw crosshairs
        ui_overlay = np.zeros(cropped.shape, cropped.dtype)
        clicked_y = int(y_off - y_range[0])
        clicked_x = int(x_off - x_range[0])
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

        # composite = cv2.bitwise_xor(img, ui_overlay)
        composite = cv2.addWeighted(cropped, 1.0, ui_overlay, 0.5, 1.0)

        self.lbl_zoom.setPixmap(QPixmap.fromImage(
            QImage(composite.data, w, h, w, QImage.Format.Format_Grayscale8)
        ))

    # compute a window that is `opening` wide that tries its best to center around `center`, but does not exceed [0, max)
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
            x * (self.max_res[0] / res_x) / PIX_PER_UM + self.x_min_mm * 1000,
            y * (self.max_res[1] / res_y) / PIX_PER_UM + self.y_min_mm * 1000
        )
    def um_to_pix_absolute(self, um, cur_res):
        (x_um, y_um) = um
        (res_x, res_y) = cur_res
        return (
            int((x_um - self.x_min_mm * 1000) * PIX_PER_UM),
            int((y_um - self.y_min_mm * 1000) * PIX_PER_UM)
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

    app = QApplication(sys.argv)
    w = MainWindow(args.name)

    schema = Schema()

    # This will read in a schema if it exists, otherwise schema will be empty
    # Schema is saved in a separate routine, overwriting the existing file at that point.
    try:
        schema.read(Path("raw/" + args.name + "/db.json"))
        w.load_schema(schema)
    except FileNotFoundError:
        w.new_schema(args, schema) # needs full set of args because we need to know max extents
        schema.overwrite(Path("raw/" + args.name + "/db.json"))

    w.finalize_ui_load()
    w.show()

    # this should cause all the window parameters to compute to the actual displayed size,
    # versus the mock sized used during object initialization
    w.updateGeometry()
    w.resizeEvent(None)

    # run the application. execution blocks at this line, until app quits
    app.exec_()

if __name__ == "__main__":
    main()
