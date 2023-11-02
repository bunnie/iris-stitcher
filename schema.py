import json
from prims import Rect, Point, ROUNDING
import logging
from scipy.spatial import distance
import numpy as np
import math
from pathlib import Path
import cv2

# derived from reference image "full-H"
# NOTE: this may change with improvements in the microscope hardware.
# be sure to re-calibrate after adjustments to the hardware.
PIX_PER_UM = 3535 / 370
X_RES = 3840
Y_RES = 2160

SCHEMA_VERSION = 1

# This is just for documentation purposes
sample_schema = {
    'version': SCHEMA_VERSION,
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
                'norm_a' : 0.0,
                'norm_b' : 255.0,
                'norm_method' : 'MINMAX'
            }
        },
        # more 'tile' objects
    ],
    'overlaps': [
        {'by_layer' : {
            'layer_list' : [],       # list of overlapping layer indices; need to compute intersection of data.
            'algorithm' : 'average', # what algorithm to use on overlapping data, e.g. 'overlay', 'average', 'gradient', etc...
            'args' : [],             # additional arguments to the algorithm
        }},
    ]
}

class Schema():
    def __init__(self):
        self.schema = {
            'version' : SCHEMA_VERSION,
            'tiles' : {},
            'overlaps' : {},
        }
        self.auto_index = int(10000)
        self.coords_mm = []
        self.zoom_cache = []
        self.path = None

    def read(self, path):
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
            # finalize extents
            self.finalize()
            return True

    def overwrite(self):
        with open(self.path / Path('db.json'), 'w+') as config:
            config.write(json.dumps(self.schema, indent=2))

    def zoom_cache_clear(self):
        self.zoom_cache = []

    def zoom_cache_insert(self, layer, tile, img):
        self.zoom_cache += [(layer, tile, img)]

    # Takes as an argument the Path to the file added.
    def add_tile(self, fpath, a=0.0, b=255.0, method='MINMAX'):
        self.schema['tiles'][str(self.auto_index)] = {
            'file_name' : fpath.stem,
            'offset' : [0.0, 0.0],
            'norm_a' : a,
            'norm_b' : b,
            'norm_method' : method,
        }
        self.auto_index += 1

        metadata = Schema.meta_from_fname(fpath.stem)
        self.coords_mm += [(metadata['x'], metadata['y'])]

    # Recomputes the overall extents of the image
    def finalize(self, max_x=None, max_y=None):
        coords = np.unique(self.coords_mm, axis=0)

        # Find the "lower left" corner. This is done by computing the Euclidean distance
        # from all the points to a point at "very lower left", i.e. -100, -100
        dists = []
        for p in coords:
            dists += [np.linalg.norm(p - [-100, -100])]
        ll_centroid = coords[dists.index(min(dists))]
        ur_centroid = coords[dists.index(max(dists))]
        logging.info(f"Raw data: Lower-left coordinate: {ll_centroid}; upper-right coordinate: {ur_centroid}")

        if max_x:
            coords = [c for c in coords if c[0] <= ll_centroid[0] + max_x]
        if max_y:
            coords = [c for c in coords if c[1] <= ll_centroid[1] + max_y]

        if max_x is not None or max_y is not None:
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

        # create a list of x-coordinates
        self.coords = coords
        self.x_list = np.unique(np.rot90(coords)[1])
        self.y_list = np.unique(np.rot90(coords)[0])

        self.x_min_mm = self.ll_frame[0]
        self.y_min_mm = self.ll_frame[1]

    def closest_tile_to_coord_mm(self, coord_um):
        distances = distance.cdist(self.coords_mm, [(coord_um[0] / 1000, coord_um[1] / 1000)])
        closest = self.coords_mm[np.argmin(distances)]
        return closest

    def sorted_tiles(self):
        return sorted(self.schema['tiles'].items())

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
            o = t['offset']
            t['offset'] = [o[0] + x, o[1] + y]
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
                    img = cv2.normalize(img, None, alpha=float(tile['norm_a']), beta=float(tile['norm_b']), norm_type=cv2.NORM_MINMAX)
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
            if tile['norm_method'] == 'MINMAX':
                method = cv2.NORM_MINMAX
            else:
                logging.error("Unsupported normalization method in schema")
            img = cv2.normalize(img, None, alpha=float(tile['norm_a']), beta=float(tile['norm_b']), norm_type=method)
            if avg_image is None:
                avg_image = img
            else:
                avg_image = cv2.addWeighted(avg_image, 0.5, img, 0.5, 0.0)
        return avg_image

    def get_image_from_tile(self, tile):
        meta = Schema.meta_from_tile(tile)
        if meta['r'] >= 0:
            img = cv2.imread(str(self.path / Path(tile['file_name'] + ".png")), cv2.IMREAD_GRAYSCALE)
            if tile['norm_method'] == 'MINMAX':
                method = cv2.NORM_MINMAX
            else:
                logging.error("Unsupported normalization method in schema")
            img = cv2.normalize(img, None, alpha=tile['norm_a'], beta=tile['norm_b'], norm_type=method)
            return img
        else:
            # we're dealing with an average
            return self.average_image_from_tile(tile)

    # Not sure if I'm doing the rounding correctly here. I feel like
    # this can end up in a situation where the w/h is short a pixel.
    @staticmethod
    def rect_mm_from_center(coord: Point):
        t_center = coord
        w_mm = (X_RES / PIX_PER_UM) / 1000
        h_mm = (Y_RES / PIX_PER_UM) / 1000
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

    def anchor_layer_index(self):
        return max(self.schema['tiles'].keys())

    def swap_layers(self, a, b):
        temp = self.schema['tiles'][str(a)]
        self.schema['tiles'][str(a)] = self.schema['tiles'][str(b)]
        self.schema['tiles'][str(b)] = temp

    @staticmethod
    def meta_from_fname(fname):
        metadata = {}
        items = fname.split('_')
        for i in items:
            metadata[i[0]] = float(i[1:])
        # r_um is the bounding rectangle of the tile in absolute um
        metadata['r_um'] = Rect(
            Point(metadata['x'] * 1000 - X_RES / 2.0 / PIX_PER_UM, metadata['y'] * 1000 - Y_RES / 2.0 / PIX_PER_UM),
            Point(metadata['x'] * 1000 + X_RES / 2.0 / PIX_PER_UM, metadata['y'] * 1000 + Y_RES / 2.0 / PIX_PER_UM)
        )
        return metadata

    @staticmethod
    def meta_from_tile(tile):
        return Schema.meta_from_fname(tile['file_name'])