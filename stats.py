import argparse
import logging
import json
import statistics
import matplotlib.pyplot as plt

PIEZO_UM_PER_LSB = 0.007425

def meta_from_fname(fname):
    metadata = {}
    items = fname.split('_')
    for i in items:
        metadata[i[0]] = float(i[1:])
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Extract stitching stats")
    parser.add_argument(
        "--loglevel", required=False, help="set logging level (INFO/DEBUG/WARNING/ERROR)", type=str, default="INFO",
    )
    parser.add_argument(
        "--name", required=False, help="database file name", default='raw/marvell-10x/db.json'
    )
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    with open(args.name, 'r') as config:
        schema = json.loads(config.read())

        if False:
            offset_x = []
            offset_y = []
            for layer, tile in schema['tiles'].items():
                (x, y) = tile['offset']
                offset_x += [x]
                offset_y += [y]

            logging.info(f"Mean offset: {statistics.mean(offset_x):0.3f}, {statistics.mean(offset_y):0.3f}")
            logging.info(f"Stddev: {statistics.stdev(offset_x):0.3f}, {statistics.stdev(offset_y):0.3f}")
            logging.info(f"Max: {max([abs(max(offset_x)), abs(min(offset_x))]):0.3f}, {max([abs(max(offset_y)), abs(min(offset_y))]):0.3f}")
        else:
            x = []
            y = []
            z = []
            for layer, tile in schema['tiles'].items():
                meta = meta_from_fname(tile['file_name'])
                x += [meta['x']]
                y += [meta['y']]
                z += [meta['z'] - meta['p'] * PIEZO_UM_PER_LSB / 1000.0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            plt.show()

if __name__ == "__main__":
    main()
