import json
from prims import Rect, Point, ROUNDING

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

    def read(self, path):
        with open(path, 'r') as config:
            self.schema = json.loads(config.read())

    def overwrite(self, path):
        with open(path, 'w+') as config:
            config.write(json.dumps(self.schema, indent=2))

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

    def sorted_tiles(self):
        return sorted(self.schema['tiles'].items())

    def get_tile_by_coordinate(self, coord):
        for (layer, t) in self.schema['tiles'].items():
            md = self.meta_from_fname(t['file_name'])
            if round(md['x'], ROUNDING) == round(coord[0], ROUNDING) \
                and round(md['y'], ROUNDING) == round(coord[1], ROUNDING):
                return (layer, t)

        return (None, None)
    
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
        return metadata
    
    @staticmethod
    def meta_from_tile(tile):
        return Schema.meta_from_fname(tile['file_name'])