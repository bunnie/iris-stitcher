import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2
import sys

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QTimer, QSignalBlocker, Qt, QRect
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QDesktopWidget, QCheckBox, QMessageBox, QMainWindow, QPushButton, QComboBox, QSlider, QGroupBox, QGridLayout, QBoxLayout, QHBoxLayout, QVBoxLayout, QMenu, QAction

# derived from reference image "full-H"
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

UI_MAX_WIDTH = 2000
UI_MAX_HEIGHT = 2000

class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setMinimumSize(1000, 1000)
        self.timer = QTimer(self)

        # Setup widget layouts
        self.lbl_overview = QLabel()
        self.lbl_zoom = QLabel()
        v_preview = QVBoxLayout()
        v_preview.addWidget(self.lbl_overview)
        v_preview.addWidget(self.lbl_zoom)
        v_widget = QWidget()
        v_widget.setLayout(v_preview)
        self.v_preview = v_preview

        grid_main = QGridLayout()
        grid_main.setRowStretch(0, 10) # video is on row 0, have it try to be as big as possible
        grid_main.addWidget(v_widget)
        w_main = QWidget()
        w_main.setLayout(grid_main)
        self.setCentralWidget(w_main)

        self.timer.timeout.connect(self.onTimer)

        # Index and load raw image data
        raw_image_path = Path("raw/" + args.name)
        files = [file for file in raw_image_path.glob('*.png') if file.is_file()]

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

        coords = np.unique(np.array(centroids), axis=0)

        # Find the "lower left" corner. This is done by computing the euclidian distance
        # from all the points to a point at "very lower left", i.e. -100, -100
        dists = []
        for p in coords:
            dists += [np.linalg.norm(p - [-100, -100])]
        ll = coords[dists.index(min(dists))]
        ur = coords[dists.index(max(dists))]
        logging.info(f"Raw data: Lower-left coordinate: {ll}; upper-right coordinate: {ur}")

        if args.max_x:
            coords = [c for c in coords if c[0] < args.max_x]
        if args.max_y:
            coords = [c for c in coords if c[1] < args.max_y]

        if args.max_x is not None or args.max_y is not None:
            coords = np.array(coords)
            # redo the ll/ur computations
            dists = []
            for p in coords:
                dists += [np.linalg.norm(p - [-100, -100])]
            ll = coords[dists.index(min(dists))]
            ur = coords[dists.index(max(dists))]
            logging.info(f"Reduced data: Lower-left coordinate: {ll}; upper-right coordinate: {ur}")

        # Determine total area of imaging centroid
        x_mm_centroid = ur[0] - ll[0]
        y_mm_centroid = ur[1] - ll[1]
        # Determine absolute imaging area in pixels based on pixels/mm and image size
        # X_RES, Y_RES added because we have a total of one frame size surrounding the centroid
        x_res = int(math.ceil(x_mm_centroid * 1000 * PIX_PER_UM + X_RES))
        y_res = int(math.ceil(y_mm_centroid * 1000 * PIX_PER_UM + Y_RES))
        logging.info(f"Final image resolution is {x_res}x{y_res}")

        canvas = np.zeros((y_res, x_res), dtype=np.uint8)

        # Build the explorer window
        if y_res > x_res:
            height = UI_MAX_HEIGHT
            width = (UI_MAX_HEIGHT / y_res) * x_res
        else:
            width = UI_MAX_WIDTH
            height = (UI_MAX_WIDTH / x_res) * y_res

        # create a list of x-coordinates
        x_list = np.unique(coords[:, 0])

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
                img = get_image(files, c, r=2)
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

        #bounds = v_preview.geometry()
        #scaled = np.zeros((bounds.width(), bounds.height()), dtype=np.uint8)
        scaled = cv2.resize(canvas, (int(x_res * (UI_MAX_HEIGHT / y_res)), UI_MAX_HEIGHT))
        height, width = scaled.shape
        bytesPerLine = 1 * width
        qImg = QImage(scaled.data, width, height, bytesPerLine, QImage.Format.Format_Grayscale8)
        self.lbl_overview.setPixmap(QPixmap.fromImage(qImg))

        # Features to implement:
        #  - Clickable tiles that zoom into the source image
        #  - Exploring the source image with X/Y lines that show intensity vs position
        #  - Simple test to take the image reps and try to align them and see if quality improves
        #  - Feature extraction on images, showing feature hot spots
        #  - Some sort of algorithm that tries to evaluate "focused-ness"


    def onTimer(self):
        print("tick")

def get_image(files, coord, r):
    img_re = re.compile('x([0-9.\-]*)_y([0-9.\-]*)_.*_r([\d*])')
    for file in files:
        match = img_re.match(file.stem).groups()
        if len(match) == 3:
            if coord[0] == float(match[0]) and coord[1] == float(match[1]) and r == int(match[2]):
                logging.info(f"loading {str(file)}")
                return cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
    logging.error(f"Requested file was not found at {coord}, r={r}")
    return None

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
    w = MainWindow(args)
    w.show()

    # run the application. execution blocks at this line, until app quits
    app.exec_()

if __name__ == "__main__":
    main()
