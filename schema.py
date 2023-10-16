import json

SCHEMA_VERSION = 1

# This is just for documentation purposes
sample_schema = {
    'version': SCHEMA_VERSION,
    'tiles': [
        {
            0 : # layer draw order. Draws from low to high, so the lower number is on the 'bottom' of two overlapping tiles.
                # Negative numbers are allowed.
                # the smallest layer number is the "anchor" layer. Its offset should always [0,0],
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
        self.schema['tiles'][self.auto_index] = {
            'file_name' : fpath.stem,
            'offset' : [0.0, 0.0],
            'norm_a' : a,
            'norm_b' : b,
            'norm_method' : method,
        }
        self.auto_index += 1

    def sorted_tiles(self):
        return sorted(self.schema['tiles'].items())

    @staticmethod
    def parse_meta(fname):
        metadata = {}
        items = fname.split('_')
        for i in items:
            metadata[i[0]] = float(i[1:])
        return metadata
