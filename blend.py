import numpy as np
import cv2
from schema import Schema
import logging

BLEND_STRENGTH = 5 # 5 is default from sample program

def blend(self):
    sorted_tiles = self.schema.sorted_tiles()
    canvas = np.zeros((self.schema.max_res[1], self.schema.max_res[0]), dtype=np.uint8)

    # now read in the images
    x_list = []
    y_list = []

    for (_index, tile) in sorted_tiles:
        metadata = Schema.meta_from_fname(tile['file_name'])
        (x, y) = self.um_to_pix_absolute(
            (float(metadata['x']) * 1000 + float(tile['offset'][0]),
            float(metadata['y']) * 1000 + float(tile['offset'][1]))
        )
        # move center coordinate to top left
        x -= Schema.X_RES / 2
        y -= Schema.Y_RES / 2

        x_list += [int(x)]
        y_list += [int(y)]

    # partition x and y space and step along max_res chunks
    MAX_RES = 15000
    for x_range in range(min(x_list), max(x_list) + 1, MAX_RES):
        for y_range in range(min(y_list), max(y_list) + 1, MAX_RES):
            corners = []
            images = []
            masks = []
            border_x = x_range + MAX_RES
            border_y = y_range + MAX_RES
            # load in just the subset in our sector
            for (layer, tile) in sorted_tiles:
                metadata = Schema.meta_from_fname(tile['file_name'])
                (x, y) = self.um_to_pix_absolute(
                    (float(metadata['x']) * 1000 + float(tile['offset'][0]),
                    float(metadata['y']) * 1000 + float(tile['offset'][1]))
                )
                # move center coordinate to top left
                x -= Schema.X_RES / 2
                y -= Schema.Y_RES / 2

                if x_range <= x and x < x_range + MAX_RES and \
                y_range <= y and y < y_range + MAX_RES:
                    # The image needs to be RGB 8-bit per channel for the cv2 blending algorithm
                    logging.info(f"Loading {tile}")
                    img = self.schema.get_image_from_layer(layer)
                    images += [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)]
                    # the mask is 255 where pixels should be copied into the final mosaic canvas. In this case, we
                    # want to overlay the full image every time, so the mask is easy.
                    zmask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
                    masks += [zmask]
                    # the corner is the top left corner of where the image should go after alignment
                    corners += [(int(x), int(y))]
                    # the actual frames captured by this aren't exactly equal to x_range or y_range, so capture them as border values
                    if x <= border_x:
                        border_x = int(x)
                    if y <= border_y:
                        border_y = int(y)

            # this computes the size of the resulting blended image
            dst_sz = cv2.detail.resultRoi(corners=corners, images=images)
            # set up the blender algorithm. This case uses the Burt & Adelson 1983 multiresolution
            # spline algorithm (gaussian/laplacian pyramids) with some modern refinements that
            # haven't been explicitly documented by opencv.
            blender = cv2.detail_MultiBandBlender(try_gpu=1) # GPU is wicked fast - the computation is much faster than reading in the data
            # I *think* this sets how far the blending seam should go from the edge.
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * BLEND_STRENGTH / 100
            # I read "bands" as basically how deep you want the pyramids to go
            blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            # Allocates memory for the final image
            blender.prepare(dst_sz)

            # Feed the images into the blender itself
            logging.info(f"Blending range {x_range}:{x_range+MAX_RES}, {y_range}:{y_range+MAX_RES}...")
            for (img, zmask, corner) in zip(images, masks, corners):
                blender.feed(img, zmask, corner)

            # Actual computational step
            sector_rgb, sector_mask = blender.blend(None, None)
            # The result is a uint16 RGB image: some magic happens to prevent precision loss. This is good.
            # Re-normalize and convert to gray scale.
            sector = cv2.normalize(sector_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow('blended', canvas)
            sector = cv2.cvtColor(sector, cv2.COLOR_RGB2GRAY)

            # deal with possible negative offsets of x/y on the canvas
            SCALE = 0.05
            w = sector.shape[1]
            h = sector.shape[0]
            if border_x < 0:
                w = w + border_x
                x_src = -border_x
                border_x = 0
            else:
                x_src = 0
            if border_y < 0:
                h = h + border_y
                y_src = -border_y
                border_y = 0
            else:
                y_src = 0
            if border_y + h > canvas.shape[0]:
                h = canvas.shape[0] - border_y
            if border_x + w > canvas.shape[1]:
                w = canvas.shape[1] - border_x

            #cv2.imshow("sector", cv2.resize(sector, None, None, SCALE, SCALE))
            #cv2.imshow("sector_mask", cv2.resize(sector_mask, None, None, SCALE, SCALE))
            #cv2.imshow("canvas", cv2.resize(canvas, None, None, SCALE, SCALE))
            #cv2.waitKey()

            # This will lead to *a* seam, but we can't blend images bigger than a certain size so...
            cv2.copyTo(
                sector[
                    int(y_src) : int(y_src + h),
                    int(x_src) : int(x_src + w)
                ],
                sector_mask[
                    int(y_src) : int(y_src + h),
                    int(x_src) : int(x_src + w),
                ],
                canvas[
                    int(border_y) : int(border_y + h),
                    int(border_x) : int(border_x + w)
                ]
            )
            #cv2.imshow("canvas", cv2.resize(canvas, None, None, SCALE, SCALE))
            #cv2.waitKey()

    self.overview = canvas
    self.overview_dirty = False
    self.rescale_overview()