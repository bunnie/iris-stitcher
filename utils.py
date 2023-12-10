import cv2

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
#
# This routine will attempt to take as much as `img` and copy it onto canvas, clipping `img`
# where it would not fit onto canvas, at the desired `x`, `y` offsets. If `x` or `y` are negative,
# the image copy will start at an offset that would correctly map the `img` pixels into the
# available canvas area
def safe_image_broadcast(img, canvas, x, y):
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
    canvas[
        y : y + h,
        x : x + w
    ] = img[
        y_src : y_src + h,
        x_src : x_src + w
    ]