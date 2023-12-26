import cv2
import numpy as np
from math import log2, ceil

def make_gaussian_pyramid(base, levels):
    g = base.copy()
    pyramid = [g]
    for i in range(levels):
        g = cv2.pyrDown(g)
        pyramid += [g]
    return pyramid

def make_laplacian_from_gaussian(gaussian):
    lp = [gaussian[-1]]
    for i in range(len(gaussian) - 1, 0, -1):
        ge = cv2.pyrUp(gaussian[i])
        l = cv2.subtract(gaussian[i-1], ge)
        lp += [l]
    return lp

def square_image(img, pad=0):
    # square up an image to the nearest power of 2
    max_dim = max(img.shape[0], img.shape[1])
    max_dim = 2**ceil(log2(max_dim))
    sq_canvas = np.full((max_dim, max_dim), pad, dtype=np.uint8)
    # Calculate the position to paste the non-square image in the center
    x_offset = (sq_canvas.shape[1] - img.shape[1]) // 2
    y_offset = (sq_canvas.shape[0] - img.shape[0]) // 2

    # Paste the non-square image in the center of the square canvas
    sq_canvas[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = img
    return sq_canvas, (x_offset, y_offset)

def composite_gaussian_pyramid(pyramid):
    rows, cols = pyramid[0].shape
    # determine the total number of rows and columns for the composite
    composite_rows = max(rows, sum(p.shape[0] for p in pyramid[1:]))
    composite_cols = cols + pyramid[1].shape[1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.uint8)

    # store the original to the left
    composite_image[:rows, :cols] = pyramid[0]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    return composite_image

def composite_laplacian_pyramid(pyramid):
    rows, cols = pyramid[-1].shape
    # determine the total number of rows and columns for the composite
    composite_rows = max(rows, sum(p.shape[0] for p in pyramid[:-1]))
    composite_cols = cols + pyramid[-2].shape[1]
    composite_image = np.zeros((composite_rows, composite_cols),
                            dtype=np.uint8)

    # store the original to the left
    composite_image[:rows, :cols] = pyramid[-1]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in reversed(pyramid[:-1]):
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    return composite_image


# https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        assert img_padded.shape[:2] == (height_max, width_max)
        images_padded.append(img_padded)

    return images_padded

# `img`: image to copy onto canvas
# `canvas`: destination for image copy
# `x`, `y`: top left corner coordinates of canvas destination (may be unsafe values)
# `mask`: optional mask, must have dimensions identical to `canvas`. Used
#    to track what regions of the canvas has valid data for averaging. A non-zero value means
#    the canvas data has not been updated, a zero value means it has image data.
#    1. Zero, overlapping regions of mask and `canvas` are averaged with the incoming `img`
#    2. Non-zero regions of mask and `canvas` are updated with the incoming `img` without averaging
#    3. `mask` is updated with non-zero values to record where pixels have been updated on the `canvas`.
#
# This routine will attempt to take as much as `img` and copy it onto canvas, clipping `img`
# where it would not fit onto canvas, at the desired `x`, `y` offsets. If `x` or `y` are negative,
# the image copy will start at an offset that would correctly map the `img` pixels into the
# available canvas area
def safe_image_broadcast(img, canvas, x, y, result_mask=None):
    SCALE = 0.05
    w = img.shape[1]
    h = img.shape[0]
    x = int(x)
    y = int(y)
    if y > canvas.shape[0] or x > canvas.shape[1]:
        # destination doesn't even overlap the canvas
        return
    if x < 0:
        w = w + x
        x_src = -x
        x = 0
    else:
        x_src = 0
    if y < 0:
        h = h + y
        y_src = -y
        y = 0
    else:
        y_src = 0
    if y + h > canvas.shape[0]:
        h = canvas.shape[0] - y
    if x + w > canvas.shape[1]:
        w = canvas.shape[1] - x
    if result_mask is None:
        canvas[
            y : y + h,
            x : x + w
        ] = img[
            y_src : y_src + h,
            x_src : x_src + w
        ]
    if result_mask is not None:
        if False:
            # copy over pixels that did not exist before
            cv2.copyTo(
                img[
                    y_src : y_src + h,
                    x_src : x_src + w
                ],
                # non-zero values are to be copied
                result_mask[
                    y : y + h,
                    x : x + w,
                ],
                canvas[
                    y : y + h,
                    x : x + w
                ]
            )

            # build a mask that indicates the overlapping region that requires blending
            # incoming_mask is the mask of just the incoming image
            # on result_mask:   0 = image data, 1 = uninit region
            # on incoming_mask: 0 = image data, 1 = uninit region
            incoming_mask = np.ones(canvas.shape, dtype=np.uint8)
            incoming_mask[
                y : y + h,
                x : x + w
            ] = np.zeros((h, w), dtype=np.uint8)
            blend_mask_u8 = incoming_mask | result_mask
            binary_mask = blend_mask_u8 != 0
            blend_mask = np.full_like(blend_mask_u8, 1)
            blend_mask[binary_mask] = 0
            cv2.imshow("mask",
                cv2.normalize(
                    cv2.resize(blend_mask, None, None, SCALE, SCALE),
                    None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
            )

            # track which pixels have been updated by setting them to 0
            updated_region = np.zeros((h, w), dtype=np.uint8)
            result_mask[
                y : y + h,
                x : x + w
            ] = updated_region

            # expand the image to the full size of the canvas
            src_img = np.zeros(canvas.shape, dtype=np.uint8)
            src_img[
                y : y + h,
                x : x + w
            ] = img[
                y_src : y_src + h,
                x_src : x_src + w
            ]

            # square up the images to the nearest power of 2, a prereq for pyrimidal decompositions
            sq_canvas, offsets = square_image(canvas)
            sq_mask, _offsets = square_image(blend_mask, pad=0)
            sq_src, _offsets = square_image(src_img)

            # compute gaussian pyramid for canvas
            # figure out depth of pyramid
            sq_h, sq_w = sq_canvas.shape
            pyr_depth = round(max(log2(sq_w), log2(sq_h)))

            gp_canvas = make_gaussian_pyramid(sq_canvas, pyr_depth)
            # cv2.imshow("gaussian canvas", cv2.resize(
            #     composite_gaussian_pyramid(gp_canvas), None, None, SCALE, SCALE
            # ))
            gp_src = make_gaussian_pyramid(sq_src, pyr_depth)
            # cv2.imshow("gaussian source", cv2.resize(
            #     composite_gaussian_pyramid(gp_src), None, None, SCALE, SCALE
            # ))
            gp_mask = make_gaussian_pyramid(sq_mask, pyr_depth)
            # cv2.imshow("gaussian mask", cv2.resize(
            #     composite_gaussian_pyramid(gp_mask), None, None, SCALE, SCALE
            # ))

            lp_canvas = make_laplacian_from_gaussian(gp_canvas)
            lp_src = make_laplacian_from_gaussian(gp_src)
            # cv2.imshow("laplacian source", cv2.normalize(cv2.resize(
            #     composite_laplacian_pyramid(lp_src), None, None, SCALE, SCALE
            # ), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            # lp_mask = make_laplacian_from_gaussian(gp_mask)
            # cv2.imshow("laplacian mask", cv2.normalize(cv2.resize(
            #     composite_laplacian_pyramid(lp_mask), None, None, SCALE, SCALE
            # ), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

            # now merge the pyramids according to the mask
            cv2.imshow("laplacian before merge", cv2.normalize(cv2.resize(
                composite_laplacian_pyramid(lp_canvas), None, None, SCALE, SCALE
            ), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
            stack = []
            for dest, src, mask in zip(lp_canvas, lp_src, reversed(gp_mask)):
                if True:
                    cv2.copyTo(
                        src, mask, dest
                    )
                    stack += [dest]
                else: # these should be equivalent, but cv2.copyTo is much faster...
                    blended_layer = (
                        src * mask +
                        dest * (1 - mask)
                    )
                    stack += [blended_layer]
            cv2.imshow("laplacian after merge", cv2.normalize(cv2.resize(
                composite_laplacian_pyramid(lp_canvas), None, None, SCALE, SCALE
            ), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))

            # reconstruct
            sq_canvas = stack[0]
            for i in range(1, pyr_depth + 1):
                sq_canvas = cv2.pyrUp(sq_canvas)
                sq_canvas = cv2.add(sq_canvas, stack[i])

            before = cv2.resize(canvas.copy(), None, None, SCALE, SCALE)
            # return to original aspect ratio
            canvas = sq_canvas[
                offsets[1] : offsets[1] + canvas.shape[0],
                offsets[0] : offsets[0] + canvas.shape[1]
            ]
            after = cv2.resize(canvas, None, None, SCALE, SCALE)
            b_a = np.hstack((before, after))
            cv2.imshow("before/after", b_a)
            cv2.waitKey()
            return canvas, result_mask
        else:
            assert canvas.shape[0] == result_mask.shape[0] and canvas.shape[1] == result_mask.shape[1], "canvas and result_mask should have identical sizes"
            # averages everything, including areas that didn't have an image before
            avg = cv2.addWeighted(
                canvas[
                    y : y + h,
                    x : x + w
                ],
                0.5,
                img[
                    y_src : y_src + h,
                    x_src : x_src + w
                ],
                0.5,
                0.0
            )
            canvas[
                y : y + h,
                x : x + w
            ] = avg

            # fixup the areas that were averaged with black by copying over the source pixels
            cv2.copyTo(
                img[
                    y_src : y_src + h,
                    x_src : x_src + w
                ],
                result_mask[
                    y : y + h,
                    x : x + w,
                ],
                canvas[
                    y : y + h,
                    x : x + w
                ]
            )

            updated_region = np.zeros((h, w), dtype=np.uint8)
            result_mask[
                y : y + h,
                x : x + w
            ] = updated_region
            return canvas, result_mask

# move `img` by `x`, `y` and return the portion of `img` that remains within
# the original dimensions of `img`
def translate_and_crop(img, x, y):
    x_max = img.shape[1]
    y_max = img.shape[0]

    if x >= 0:
        if x < x_max:
            x_min = x
        else:
            return None
    else:
        if x + x_max > 0:
            x_min = 0
            x_max = x + x_max
        else:
            return None

    if y >= 0:
        if y < y_max:
            y_min = y
        else:
            return None
    else:
        if y + y_max > 0:
            y_min = 0
            y_max = y + y_max

    return img[
        y_min : y_max,
        x_min : x_max
    ]