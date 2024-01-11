import json
from prims import Rect, Point, ROUNDING
import logging
from scipy.spatial import distance
import numpy as np
import math
from pathlib import Path
import cv2
import datetime

from utils import pad_images_to_same_size

class Schema():
    X_RES = 3840
    Y_RES = 2160
    SCHEMA_VERSION = "1.0.1"
    # derived from reference image "full-H"
    # NOTE: this may change with improvements in the microscope hardware.
    # be sure to re-calibrate after adjustments to the hardware.
    PIX_PER_UM_20X = 3535 / 370 # 20x objective
    PIX_PER_UM_5X = 2350 / 1000 # 5x objective, 3.94 +/- 0.005 ratio to 20x
    PIX_PER_UM_10X = 3330 / 700 # 10x objective, ~4.757 pix/um
    PIX_PER_UM = None
    LAPLACIAN_WINDOW_20X = 27 # 20x objective
    LAPLACIAN_WINDOW_10X = 5 # needs tweaking
    LAPLACIAN_WINDOW_5X = 11 # 5x objective (around 7-11 seems to be a good area?)
    LAPLACIAN_WINDOW = None
    FILTER_WINDOW_20X = 31 # guess, needs tweaking
    FILTER_WINDOW_10X = 7
    FILTER_WINDOW_5X = 5 # guess, needs tweaking
    FILTER_WINDOW = None
    NOM_STEP_20x = 0.1
    NOM_STEP_10x = 0.3
    NOM_STEP_5x = 0.5
    NOM_STEP = None
    @staticmethod
    def set_mag(mag):
        if mag == 20:
            Schema.PIX_PER_UM = Schema.PIX_PER_UM_20X
            Schema.LAPLACIAN_WINDOW = Schema.LAPLACIAN_WINDOW_20X
            Schema.NOM_STEP = Schema.NOM_STEP_20x
            Schema.FILTER_WINDOW = Schema.FILTER_WINDOW_20X
        elif mag == 5:
            Schema.PIX_PER_UM = Schema.PIX_PER_UM_5X
            Schema.LAPLACIAN_WINDOW = Schema.LAPLACIAN_WINDOW_5X
            Schema.NOM_STEP = Schema.NOM_STEP_5x
            Schema.FILTER_WINDOW = Schema.FILTER_WINDOW_5X
        elif mag == 10:
            Schema.PIX_PER_UM = Schema.PIX_PER_UM_10X
            Schema.LAPLACIAN_WINDOW = Schema.LAPLACIAN_WINDOW_10X
            Schema.NOM_STEP = Schema.NOM_STEP_10x
            Schema.FILTER_WINDOW = Schema.FILTER_WINDOW_10X
        else:
            logging.error(f"Unhandled magnification parameter: {mag}")
    @staticmethod
    def set_laplacian(value):
        Schema.LAPLACIAN_WINDOW = value
    @staticmethod
    def set_filter(value):
        Schema.FILTER_WINDOW = value

    def __init__(self, use_cache=True):
        self.schema = {
            'version' : Schema.SCHEMA_VERSION,
            'tiles' : {},
            'overlaps' : {},
        }
        self.auto_index = int(10000)
        self.coords_mm = []
        self.zoom_cache = []
        self.path = None
        self.save_name = None
        self.save_type = 'png'
        self.average = False
        self.avg_qc = False
        self.use_cache = use_cache
        self.image_cache = {}
        self.max_x = None
        self.max_y = None

    def set_save_name(self, name):
        self.save_name = name

    def set_save_type(self, extension):
        self.save_type = extension.lstrip('.')

    # saves the image and the database file that was used to generate it
    def save_image(self, img, modifier=''):
        if modifier != '':
            modifier = '_' + modifier
        now = datetime.datetime.now()
        cv2.imwrite(f'{self.save_name}{modifier}_{now.strftime("%m%d%Y_%H%M%S")}.{self.save_type}', img)
        with open(Path(f'db_{self.save_name}{modifier}_{now.strftime("%m%d%Y_%H%M%S")}.json'), 'w+') as config:
            config.write(json.dumps(self.schema, indent=2))

    def read(self, path, max_x=None, max_y=None):
        self.max_x = max_x
        self.max_y = max_y
        fullpath = path / Path('db.json')
        if not fullpath.is_file():
            # For some reason the FileNotFoundError is not being propagated
            # back to the caller, I'm having to do this weird thing. Probably
            # some import has...changed the behaviors of exceptions in a way
            # I don't expect and I don't know which one it was. Fucking Python.
            return False
        with open(fullpath, 'r') as config:
            self.path = path
            self.schema = json.loads(config.read())
            # extract locations of all the tiles
            for (_layer, t) in self.schema['tiles'].items():
                metadata = Schema.meta_from_fname(t['file_name'])
                self.coords_mm += [(metadata['x'], metadata['y'])]
                # add records not present in the initial implementations of this revision
                if 'auto_error' not in t:
                    t['auto_error'] = 'invalid'
                if 'score' not in t:
                    t['score'] = -1.0
                if 'solutions' not in t:
                    t['solutions'] = 1 # leave it as a passing value if it's legacy database

            # migrate the version number, if it's old, as the prior code patches it up
            if self.schema['version'] == 1:
                self.schema['version'] = '1.0.1'

            # finalize extents
            self.finalize()
            return True

    def overwrite(self):
        logging.info(f"Saving schema to {self.path / Path('db.json')}")
        with open(self.path / Path('db.json'), 'w+') as config:
            config.write(json.dumps(self.schema, indent=2))

    def zoom_cache_clear(self):
        self.zoom_cache = []

    def zoom_cache_insert(self, layer, tile, img):
        self.zoom_cache += [(layer, tile, img)]

    # Takes as an argument the Path to the file added.
    def add_tile(self, fpath, a=0.0, b=255.0, method='MINMAX', max_x = None, max_y = None):
        metadata = Schema.meta_from_fname(fpath.stem)
        self.schema['tiles'][str(self.auto_index)] = {
            'file_name' : fpath.stem,
            'offset' : [0.0, 0.0],
            'norm_a' : a,
            'norm_b' : b,
            'norm_method' : method,
            'auto_error' : 'invalid',
            'score' : -1.0,
            'solutions' : 0,
        }
        self.auto_index += 1
        self.coords_mm += [(metadata['x'], metadata['y'])]

    def contains_layer(self, layer):
        return self.schema['tiles'].get(str(layer)) is not None

    # Recomputes the overall extents of the image
    def finalize(self):
        max_x = self.max_x
        max_y = self.max_y
        self.coords_mm = coords = np.unique(self.coords_mm, axis=0)

        # Find the "top left" corner. This is done by computing the Euclidean distance
        # from all the points to a point at "very top left", i.e. -100, -100
        dists = []
        for p in coords:
            dists += [np.linalg.norm(p - [-100, -100])]
        tl_centroid = coords[dists.index(min(dists))]
        br_centroid = coords[dists.index(max(dists))]
        logging.info(f"Raw data: Lower-left coordinate: {tl_centroid}; upper-right coordinate: {br_centroid}")

        if max_x:
            coords = [c for c in coords if c[0] <= tl_centroid[0] + max_x]
        if max_y:
            coords = [c for c in coords if c[1] <= tl_centroid[1] + max_y]

        if max_x is not None or max_y is not None:
            coords = np.array(coords)
            # redo the ll/ur computations
            dists = []
            for p in coords:
                dists += [np.linalg.norm(p - [-100, -100])]
            tl_centroid = coords[dists.index(min(dists))]
            br_centroid = coords[dists.index(max(dists))]
            logging.info(f"Reduced data: Lower-left coordinate: {tl_centroid}; upper-right coordinate: {br_centroid}")

        # note that ur, ll are the coordinates of the center of the images forming the tiles. This means
        # the actual region shown is larger, because the images extend out from the center of the images.

        # Determine total area of imaging centroid
        x_mm_centroid = br_centroid[0] - tl_centroid[0]
        y_mm_centroid = br_centroid[1] - tl_centroid[1]
        # Determine absolute imaging area in pixels based on pixels/mm and image size
        # X_RES, Y_RES added because we have a total of one frame size surrounding the centroid
        x_res = int(math.ceil(x_mm_centroid * 1000 * Schema.PIX_PER_UM + Schema.X_RES))
        y_res = int(math.ceil(y_mm_centroid * 1000 * Schema.PIX_PER_UM + Schema.Y_RES))
        logging.info(f"Final image resolution is {x_res}x{y_res}")
        # resolution of total area
        self.max_res = (x_res, y_res)

        self.tl_frame = [tl_centroid[0] - (Schema.X_RES / (2 * Schema.PIX_PER_UM)) / 1000, tl_centroid[1] - (Schema.Y_RES / (2 * Schema.PIX_PER_UM)) / 1000]
        self.br_frame = [br_centroid[0] + (Schema.X_RES / (2 * Schema.PIX_PER_UM)) / 1000, br_centroid[1] + (Schema.Y_RES / (2 * Schema.PIX_PER_UM)) / 1000]

        # create a list of x-coordinates
        self.coords = coords
        self.x_list = np.unique(np.rot90(coords)[1])
        self.y_list = np.unique(np.rot90(coords)[0])

        self.x_min_mm = self.tl_frame[0]
        self.y_min_mm = self.tl_frame[1]
        self.tl_centroid = tl_centroid
        self.br_centroid = br_centroid

        # remove filtered elements
        to_remove = []
        for (layer, tile) in self.schema['tiles'].items():
            meta = Schema.meta_from_tile(tile)
            coord = [float(meta['x']), float(meta['y'])]
            if coord not in coords:
                to_remove += [layer]
        for remove in to_remove:
            self.remove_tile(remove)

    def closest_tile_to_coord_mm(self, coord_um):
        offset_coords_mm = []
        original_mm = []
        for (_layer, t) in self.schema['tiles'].items():
            metadata = Schema.meta_from_fname(t['file_name'])
            offset_coords_mm += [(
                metadata['x'] + t['offset'][0] / 1000,
                metadata['y'] + t['offset'][1] / 1000,
            )]
            original_mm += [(metadata['x'], metadata['y'])]
        # determine distances based on actual coordinates
        distances = distance.cdist(offset_coords_mm, [(coord_um[0] / 1000, coord_um[1] / 1000)])
        # translate back to the original mm coordinates
        closest = original_mm[np.argmin(distances)]
        return closest

    def sorted_tiles(self):
        return sorted(self.schema['tiles'].items())
    def tiles(self):
        return self.schema['tiles'].items()

    def remove_tile(self, layer):
        del self.schema['tiles'][layer]

    def get_tile_by_coordinate(self, coord):
        for (layer, t) in self.schema['tiles'].items():
            md = self.meta_from_fname(t['file_name'])
            if round(md['x'], ROUNDING) == round(coord[0], ROUNDING) \
                and round(md['y'], ROUNDING) == round(coord[1], ROUNDING):
                return (layer, t)

        return (None, None)

    def adjust_offset(self, layer, x, y):
        t = self.schema['tiles'][str(layer)]
        if t is not None:
            #o = t['offset']
            #t['offset'] = [o[0] + x, o[1] + y]
            t['offset'] = [x, y]
        else:
            logging.error("Layer f{layer} not found in adjusting offset!")

        # NO! The zoom cache turns out to be references, not copies.
        if False:
            # also need to update in zoom cache
            for (index, (cache_layer, t, img)) in enumerate(self.zoom_cache):
                if layer == cache_layer:
                    o = t['offset']
                    t['offset'] = [o[0] + x, o[1] + y]
                    self.zoom_cache[index] = (cache_layer, t, img)
                    return

    def store_auto_align_result(self, layer, score, error, set_anchor=False, solutions=0):
        layer = str(layer)
        if score is None:
            # an invalid score is interpreted as a stitching error
            self.schema['tiles'][layer]['score'] = -1.0
            self.schema['tiles'][layer]['auto_error'] = 'true'
            self.schema['tiles'][layer]['solutions'] = solutions
            return
        self.schema['tiles'][layer]['score'] = score
        self.schema['tiles'][layer]['solutions'] = solutions
        if set_anchor:
            self.schema['tiles'][layer]['auto_error'] = 'anchor'
        else:
            if error:
                self.schema['tiles'][layer]['auto_error'] = 'true'
            else:
                self.schema['tiles'][layer]['auto_error'] = 'false'

    def reset_all_align_results(self):
        logging.info("Resetting all stitching scores except for the anchor")
        anchor_layer = self.anchor_layer_index()
        for (layer, t) in self.schema['tiles'].items():
            if str(layer) != str(anchor_layer):
                t['score'] = -1.0
                t['auto_error'] = 'false'
                t['offset'] = [0, 0]

    def cycle_rev(self, layer):
        if layer is None:
            logging.warn("Select a layer first")
            return None
        try:
            t = self.schema['tiles'][str(layer)]
        except:
            logging.error("Selected layer not in tile list")
            return None

        if t is not None:
            fname = t['file_name']
            # find all files in the same rev group
            f_root = fname.split('_r')[0]
            f_rev = fname.split('_r')[1]
            revs = [rev for rev in self.path.glob(f_root + '_r*.png') if rev.is_file()]
            new_rev = None
            for rev in revs:
                if str(rev.stem.split('_r')[1]) == str(int(f_rev) + 1):
                    new_rev = rev
                    break
            if new_rev is None: # as is the case if we were coming out of an average
                new_rev = revs[0]

            t['file_name'] = new_rev.stem
            # now update the cache, if appropriate
            for (index, (l, tile, img)) in enumerate(self.zoom_cache):
                if layer == l:
                    tile['file_name'] = new_rev.stem
                    img = cv2.imread(str(self.path / Path(tile['file_name'] + ".png")), cv2.IMREAD_GRAYSCALE)
                    # img = cv2.normalize(img, None, alpha=float(tile['norm_a']), beta=float(tile['norm_b']), norm_type=cv2.NORM_MINMAX)
                    self.zoom_cache[index] = (l, tile, img)
                    return new_rev.stem.split('_r')[1]

    def set_avg(self, layer):
        if layer is None:
            logging.warn("Select a layer first")
            return None
        try:
            t = self.schema['tiles'][str(layer)]
        except:
            logging.error("Selected layer not in tile list")
            return None

        if t is not None:
            fname = t['file_name']
            # find all files in the same rev group
            f_root = fname.split('_r')[0]
            revs = [rev for rev in self.path.glob(f_root + '_r*.png') if rev.is_file()]

            t['file_name'] = f_root + '_r-1'
            for (index, (l, tile, img)) in enumerate(self.zoom_cache):
                if layer == l:
                    tile['file_name'] = f_root + '_r-1'
                    img = self.average_image_from_tile(tile)
                    self.zoom_cache[index] = (l, tile, img)
                    return 'avg'

    def average_image_from_tile(self, tile):
        fname = tile['file_name']
        # find all files in the same rev group
        f_root = fname.split('_r')[0]
        revs = [rev for rev in self.path.glob(f_root + '_r*.png') if rev.is_file()]
        avg_image = None
        for rev in revs:
            img = cv2.imread(str(self.path / Path(rev.stem + ".png")), cv2.IMREAD_GRAYSCALE)
            if avg_image is None:
                avg_image = img
            else:
                if self.avg_qc:
                    logging.info(f"avg-qc: check {rev.stem + '.png'}")
                    # first align the averaged images, and check the score
                    SEARCH_SCALE = 0.9
                    FAILING_SCORE = 30.0
                    # extract the template image
                    template_rect_full = Rect(Point(0, 0), Point(Schema.X_RES, Schema.Y_RES))
                    template_rect = template_rect_full.scale(SEARCH_SCALE)
                    template_ref = template_rect.tl - template_rect_full.tl
                    template = img[
                        int(template_rect.tl.y) : int(template_rect.br.y),
                        int(template_rect.tl.x) : int(template_rect.br.x)
                    ].copy()
                    # compute ref/template normalized laplacians
                    ref_norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    template_norm = cv2.normalize(template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    ref_laplacian = cv2.Laplacian(ref_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)
                    template_laplacian = cv2.Laplacian(template_norm, -1, ksize=Schema.LAPLACIAN_WINDOW)

                    # template matching
                    METHOD = cv2.TM_CCOEFF
                    res = cv2.matchTemplate(ref_laplacian, template_laplacian, METHOD)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    match_pt = max_loc
                    res_8u = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    ret, thresh = cv2.threshold(res_8u, 224, 255, 0)

                    # find contours of candidate matches
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(res_8u, contours, -1, (0,255,0), 1)
                    #cv2.imshow('contours', res_8u)
                    #cv2.waitKey(10)

                    # use the contours and the matched point to measure the quality of the template match
                    has_single_solution = True
                    score = None
                    for index, c in enumerate(contours):
                        if hierarchy[0][index][3] == -1:
                            if cv2.pointPolygonTest(c, match_pt, False) >= 0.0: # detect if point is inside or on the contour. On countour is necessary detect cases of an exact match.
                                if score is not None:
                                    has_single_solution = False
                                score = cv2.contourArea(c)
                                logging.debug(f"countour {c} contains {match_pt} and has area {score}")
                            else:
                                # print(f"countour {c} does not contain {top_left}")
                                pass
                        else:
                            if cv2.pointPolygonTest(c, match_pt, False) > 0:
                                logging.info(f"{match_pt} is contained within a donut-shaped region. Suspect blurring error!")
                                has_single_solution = False

                    if score is not None and has_single_solution and score < FAILING_SCORE: # store the stitch if a good match was found
                        adjustment_vector_px = Point(
                            match_pt[0] - template_ref[0],
                            match_pt[1] - template_ref[1]
                        )
                        logging.debug(f"adjustment: {adjustment_vector_px}")
                        if adjustment_vector_px[0] > 1.0 or adjustment_vector_px[1] > 1.0:
                            logging.info(f"Misalignment detected: {rev.stem + '.png'} by {adjustment_vector_px}; not averaging in")
                            # this error could be "recoverable" with some offsetting tricks but the question is what do
                            # you do with the edge pixels that don't match up! we can end up with inconsistently sized
                            # base frames that violate some core principles of the downstream algorithms. Since mis-matched
                            # averages are rare, we just ignore them for now.
                        else:
                            avg_image = cv2.addWeighted(avg_image, 0.5, img, 0.5, 0.0)
                    else:
                        logging.warning(f"Skipping average of {rev.stem + '.png'}, image quality too low (score: {score})")
                        PREVIEW_SCALE = 0.3
                        debug_img = np.hstack(pad_images_to_same_size(
                            (
                                cv2.resize(img, None, None, PREVIEW_SCALE, PREVIEW_SCALE),
                                cv2.resize(avg_image, None, None, PREVIEW_SCALE, PREVIEW_SCALE)
                            )
                        ))
                        cv2.imshow('average error', debug_img)
                        cv2.waitKey(10)
                else:
                    avg_image = cv2.addWeighted(avg_image, 0.5, img, 0.5, 0.0)

        return avg_image

    def get_info_from_layer(self, layer):
        tile = self.schema['tiles'][layer]
        meta = Schema.meta_from_tile(tile)
        return (meta, tile)

    def get_image_from_layer(self, layer):
        tile = self.schema['tiles'][layer]
        meta = Schema.meta_from_tile(tile)

        if self.use_cache and layer in self.image_cache:
            return self.image_cache[layer]

        if meta['r'] >= 0:
            logging.info(f"Loading {tile}")
            img = cv2.imread(str(self.path / Path(tile['file_name'] + ".png")), cv2.IMREAD_GRAYSCALE)
            if tile['norm_method'] == 'MINMAX':
                method = cv2.NORM_MINMAX
            else:
                logging.error("Unsupported normalization method in schema")

            if self.average:
                img = self.average_image_from_tile(tile)

            if self.use_cache:
                self.image_cache[layer] = img.copy()
            return img
        else:
            # we're dealing with an average
            img = self.average_image_from_tile(tile)
            if self.use_cache:
                self.image_cache[layer] = img.copy()
            return img

    # Not sure if I'm doing the rounding correctly here. I feel like
    # this can end up in a situation where the w/h is short a pixel.
    @staticmethod
    def rect_mm_from_center(coord: Point):
        t_center = coord
        w_mm = (Schema.X_RES / Schema.PIX_PER_UM) / 1000
        h_mm = (Schema.Y_RES / Schema.PIX_PER_UM) / 1000
        return Rect(
            Point(
                round(t_center[0] - w_mm / 2, ROUNDING),
                round(t_center[1] - h_mm / 2, ROUNDING),
            ),
            Point(
                round(t_center[0] + w_mm / 2, ROUNDING),
                round(t_center[1] + h_mm / 2, ROUNDING)
            )
        )

    # This routine returns a sorted dictionary of intersecting tiles, keyed by layer draw order,
    # that intersect with `coord`. This routine does *not* adjust the intersection computation
    # by the `offset` field, so that you don't "lose" a tile as you move it over the border
    # of tiling zones.
    def get_intersecting_tiles(self, coord_mm):
        center = Point(coord_mm[0], coord_mm[1])
        rect = Schema.rect_mm_from_center(center)
        result = {}
        for (layer, t) in self.schema['tiles'].items():
            md = self.meta_from_fname(t['file_name'])
            t_center = Point(float(md['x']), float(md['y']))
            t_rect = Schema.rect_mm_from_center(t_center)
            if rect.intersection(t_rect) is not None:
                result[layer] = t

        return sorted(result.items())

    def center_coord_from_tile(self, tile):
        md = self.meta_from_tile(tile)
        return Point(float(md['x']), float(md['y']))

    def anchor_layer_index(self):
        return max(self.schema['tiles'].keys())

    def swap_layers(self, a, b):
        self.schema['tiles'][str(a)], self.schema['tiles'][str(b)] = self.schema['tiles'][str(b)], self.schema['tiles'][str(a)]

    @staticmethod
    def meta_from_fname(fname):
        metadata = {}
        items = fname.split('_')
        for i in items:
            metadata[i[0]] = float(i[1:])
        # r_um is the bounding rectangle of the tile in absolute um
        metadata['r_um'] = Rect(
            Point(metadata['x'] * 1000 - Schema.X_RES / 2.0 / Schema.PIX_PER_UM, metadata['y'] * 1000 - Schema.Y_RES / 2.0 / Schema.PIX_PER_UM),
            Point(metadata['x'] * 1000 + Schema.X_RES / 2.0 / Schema.PIX_PER_UM, metadata['y'] * 1000 + Schema.Y_RES / 2.0 / Schema.PIX_PER_UM)
        )
        return metadata

    @staticmethod
    def meta_from_tile(tile):
        return Schema.meta_from_fname(tile['file_name'])

    @staticmethod
    def fname_from_meta(meta):
        fname = ''
        for k,v in meta.items():
            if isinstance(v, (float, int)):
                if fname != '':
                    fname += '_'
                fname += str(k)
                if k == 'x' or k == 'y' or k == 'z':
                    fname += f"{v:.2f}"
                elif k == 'a':
                    fname += f"{v:.1f}"
                else:
                    fname += f"{int(v)}"
            else:
                # ignore other objects that were made, like the 'r_um' Rect
                pass
        return fname

# This is just for documentation purposes
sample_schema = {
    'version': Schema.SCHEMA_VERSION,
    'tiles': [
        {
            0 : # layer draw order. Draws from low to high, so the lower number is on the 'bottom' of two overlapping tiles.
                # Negative numbers are allowed.
                # the largest layer number is the "anchor" layer. Its offset should always [0,0],
                # and all other offsets are relative to this point. The layer number is unique,
                # duplicate layers numbers are not allowed.
            {
                'file_name' : 'none',  # name of the source file. Encodes centroid, revision, etc.
                # file name format must be as follows:
                # [path]/x[x coordinate in mm]_y[y coordinate in mm]_z[z coordinate in mm]_p[piezo parameter]_i[intensity]_t[theta of light source]_r[image revision].png
                # in some revs, an additional _a[rotational angle] parameter may be present.
                # - x, y, z are in mm.
                # - p is unitless; it corresponds to the DAC code used to drive the piezo positioner
                # - t is unitless; it corresponds to the angular setting put into the theta servo
                # - a is unitless; it corresponds to the angular setting put into the psi servo
                # - i is unitless; it corresponds to a linear brightness to peak brightness at 4095
                'offset' : [0, 0],     # offset from nominal centroid in micron
                'score' : 0.0,         # score of the auto-alignment algorithm
                'solutions' : 0,
                'auto_error' : 'invalid',  # true if there was an error during automatic alignment; invalid if alignment not yet run; false if no error; anchor if an anchor layer
                'norm_a' : 0.0,
                'norm_b' : 255.0,
                'norm_method' : 'MINMAX'
            }
        },
        # more 'tile' objects
    ],
    # this is a list of actions taken. latest actions are at the end of the list.
    'undo': [
        ('action', 0, {}) # action, layer, tile. action is a string that describes what was done; layer is a number; tile is a dictionary
    ],
    'overlaps': [
        {'by_layer' : {
            'layer_list' : [],       # list of overlapping layer indices; need to compute intersection of data.
            'algorithm' : 'average', # what algorithm to use on overlapping data, e.g. 'overlay', 'average', 'gradient', etc...
            'args' : [],             # additional arguments to the algorithm
        }},
    ]
}

