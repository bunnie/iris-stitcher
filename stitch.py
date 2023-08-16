import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2

# derived from reference image "full-H"
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

UI_MAX_WIDTH = 2000
UI_MAX_HEIGHT = 2000

class Rect():
    def __init__(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        self.lx = min([x1, x2])
        self.rx = max([x1, x2])
        self.ty = min([y1, y2])
        self.by = max([y1, y2])
    def is_hit(self, p):
        (x, y) = p
        return x >= self.lx and x <= self.rx and y >= self.ty and y <= self.by
    def bl(self):
        return (self.lx, self.by)
    def tr(self):
        return (self.rx, self.ty)
    def width(self):
        return self.rx - self.lx
    def height(self):
        return self.by - self.ty
    def translate(self, p):
        (x, y) = p
        return Rect((self.lx + x, self.by + y), (self.rx + x, self.ty + y))

class Button(Rect):
    def __init__(self, p1, p2, name):
        self.name = name
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.size = 0.5
        self.thickness = 3
        self.button_color = (255, 255, 255)
        self.text_color = (16, 16, 16)
        self.text_thickness = 1
        self.action = self.default_action
        super().__init__(p1, p2)

    @classmethod
    def from_rect(cls, rect, name):
        return cls(rect.bl(), rect.tr(), name)

    def redraw(self, window):
        cv2.rectangle(window, self.bl(), self.tr(), self.button_color, -1)
        ((width, height), _wtf) = cv2.getTextSize(self.name, self.font, self.size, self.thickness)
        cv2.putText(window, self.name,
                    (self.width() // 2 - width // 2 + self.lx,
                     self.height() // 2 + height // 2 + self.ty),
                    self.font, self.size, self.text_color, self.text_thickness
                    )
    def default_action(self):
        print(f"{self.name} clicked!")

def iris_button_hook(event, x, y, flags, param):
    param.on_event(event, x, y, flags)

class IrisWindow():
    def __init__(self, width, height):
        self.name = 'IRIS'

        self.buttons = []
        if False: # reminder code
            button_a = Button((10, 10), (110, 50), "button A")
            button_b = Button.from_rect(button_a.translate((150, 0)), "button B")
            self.buttons += [button_a, button_b]

        self.window_size = (width, height)
        cv2.namedWindow(self.name)
        cv2.resizeWindow(self.name, *self.window_size)
        self.window = np.zeros((self.window_size[1], self.window_size[0], 3), np.uint8)
        cv2.setMouseCallback(self.name, iris_button_hook, self)
        self.redraw()

    def redraw(self):
        for b in self.buttons:
            b.redraw(self.window)

    def on_event(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            logging.debug("click")
            for b in self.buttons:
                if b.is_hit((x, y)):
                    b.action()

    def show_image(self, img):
        color_img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        # fit to X, without stretching
        width = self.window.shape[1]
        height = int(color_img.shape[0] * (self.window.shape[1] / color_img.shape[1]))
        fit_image = cv2.resize(color_img, (width, height))
        self.window[0:fit_image.shape[0], 0:fit_image.shape[1]] = fit_image

    def loop(self):
        while True:
            cv2.imshow(self.name, self.window)
            self.redraw()
            key =  cv2.waitKey(1)

            if key == ord('q'):
                break

        cv2.destroyAllWindows()

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
    w = IrisWindow(int(width), int(height))

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

    w.show_image(canvas)
    w.loop()


if __name__ == "__main__":
    main()
