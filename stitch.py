import argparse
from pathlib import Path
import logging
import numpy as np
import math
import re
import cv2

from bokeh.layouts import column, layout
from bokeh.plotting import figure, show
from bokeh.palettes import grey
from bokeh.models import CrosshairTool, Span, RangeTool, BoxSelectTool, ColumnDataSource

from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from tornado import web
from bokeh.events import Tap, SelectionGeometry

# derived from reference image "full-H"
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

UI_MAX_WIDTH = 1200
UI_MAX_HEIGHT = 800

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

def get_image_filename(files, coord, r):
    img_re = re.compile('x([0-9.\-]*)_y([0-9.\-]*)_.*_r([\d*])')
    for file in files:
        match = img_re.match(file.stem).groups()
        if len(match) == 3:
            if coord[0] == float(match[0]) and coord[1] == float(match[1]) and r == int(match[2]):
                logging.info(f"Retrieving {str(file)}")
                return file
    logging.error(f"Requested file was not found at {coord}, r={r}")
    return None


def make_makedoc(args):
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

    # p = figure(width=UI_MAX_WIDTH, height=UI_MAX_HEIGHT)
    # p.x_range.range_padding = p.y_range.range_padding = 0
    # p.image(
    #     image=[canvas],
    #     x=0, y=0, dw=x_res / PIX_PER_UM, dh=y_res / PIX_PER_UM,
    #     palette=grey(256), level="image",
    #     dilate=True
    # )
    # p.grid.grid_line_width=0.5

    # #width = Span(dimension="width", line_dash="dotted", line_width=2)
    # #height = Span(dimension="height", line_dash="dotted", line_width=2)
    # p.add_tools(BoxSelectTool(description="focus area", persistent=True, ))

    # show(p)

    # Features to implement:
    #  - Clickable tiles that zoom into the source image
    #  - Exploring the source image with X/Y lines that show intensity vs position
    #  - Simple test to take the image reps and try to align them and see if quality improves
    #  - Feature extraction on images, showing feature hot spots
    #  - Some sort of algorithm that tries to evaluate "focused-ness"

    def makedoc(doc):
        p = figure(width=UI_MAX_WIDTH, height=UI_MAX_HEIGHT)
        p.x_range.range_padding = p.y_range.range_padding = 0
        p.image(
            image=[canvas],
            x=0, y=0, dw=x_res / PIX_PER_UM, dh=y_res / PIX_PER_UM,
            palette=grey(256), level="image",
            dilate=True, anchor='top_left'
        )
        p.grid.grid_line_width=0.5

        zoomed = figure(width=UI_MAX_WIDTH, height=UI_MAX_HEIGHT)
        zoomed.x_range.range_padding = p.y_range.range_padding = 0
        zoomed_holder = ColumnDataSource({'image': [],
                                        'x': [], 'y': [],
                                        'dx': [], 'dy': []})
        zoomed.image_url('image', 'x', 'y', 'dx', 'dy',
                                source=zoomed_holder,
                                anchor='top_left')
        ## region = np.zeros((UI_MAX_WIDTH, UI_MAX_HEIGHT), dtype=np.uint8)
        # zoomed.image(
        #     image=region,
        #     x=0, y=0, dw=x_res / PIX_PER_UM, dh=y_res / PIX_PER_UM,
        #     palette=grey(256), level="image",
        #     dilate=True
        # )

        def callback(event):
            print(f"boink: {event.x}, {event.y}")
            # TODO: fix coordinates based on actual loaded data offsets
            y_mm = -round(event.y / 1000, 1)
            x_mm = round(event.x / 1000, 1)
            fpath = get_image_filename(files, (x_mm, y_mm), 2)
            if fpath is not None:
                fname = 'images/' + fpath.name
                print(f"using {fname}")
                zoomed_holder.data = {'image': [fname], 'x': [
                    0], 'y': [0], 'dx': [X_RES / PIX_PER_UM], 'dy': [Y_RES / PIX_PER_UM]}
            else:
                logging.error(f"Couldn't find file at {x_mm} {y_mm}")
        p.on_event(Tap, callback)

        #bs = BoxSelectTool(description="focus area", persistent=True)
        #   bs.on_event(SelectionGeometry, callback)
        #p.add_tools(bs)

        page_content = layout([
            p, zoomed
        ])
        doc.title = 'IRIS Image Navigator'
        doc.add_root(page_content)
    print('ready!')
    return makedoc

def run_server(path='/', port=5000,
               url='http://localhost'):
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

    makedoc = make_makedoc(args)
    apps = {path: Application(FunctionHandler(makedoc))}
    server = Server(apps, port=port, allow_websocket_origin=['*'])
    server.start()
    print('Web app now available at {}:{}'.format(url, port))
    handlers = [(path + r'images/(.*)',
                 web.StaticFileHandler,
                {'path': './raw/' + args.name + '/'})]
    server._tornado.add_handlers(r".*", handlers)
    server.run_until_shutdown()


if __name__ == "__main__":
    run_server()
