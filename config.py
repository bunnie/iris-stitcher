# ------ global configs
X_RES = 3536
Y_RES = 3536
PIX_PER_UM_20X = 3535 / 370 # 20x objective
PIX_PER_UM_5X = 2350 / 1000 # 5x objective, 3.94 +/- 0.005 ratio to 20x
# Originally 3330 / 700
#   - Adjusted to 3327 / 700 on 12/19/2024 - 10x with 3840x3840 camera upgrade [bunnie]
PIX_PER_UM_10X = 3327 / 700 # 10x objective, ~4.757 pix/um
LAPLACIAN_WINDOW_20X = 27 # 20x objective
LAPLACIAN_WINDOW_10X = 5 # needs tweaking
LAPLACIAN_WINDOW_5X = 11 # 5x objective (around 7-11 seems to be a good area?)
FILTER_WINDOW_20X = 31 # guess, needs tweaking
FILTER_WINDOW_10X = 7
FILTER_WINDOW_5X = 5 # guess, needs tweaking
NOM_STEP_20x = 0.1
NOM_STEP_10x = 0.3
NOM_STEP_5x = 0.5
THUMB_SCALE = 0.125 # scaling of a thumbnail (in linear dimensions)
THUMB_THRESHOLD_PX = 2000 # threshold to use thumbnails for the overview

UI_MIN_WIDTH = 1000
UI_MIN_HEIGHT = 1000

TILES_VERSION = 1

PIEZO_UM_PER_LSB = (1/193.5) # from empirical measurements
SECULAR_PIEZO_UM_PER_LSB = 0.007425 # from datasheet numbers

# ----- configure template matching
# low scores are better. scores greater than this fail.
FAILING_SCORE = 80.0
CONTOUR_THRESH = 192 # 192 for well-focused images; 224 if the focus quality is poor
# maximum number of potential solutions before falling back to manual review
MAX_SOLUTIONS = 8
PREVIEW_SCALE = 0.3
X_REVIEW_THRESH_UM = 110.0
Y_REVIEW_THRESH_UM = 110.0
SEARCH_SCALE = 0.80  # 0.8 worked on the AW set, 0.9 if using a square template
MAX_TEMPLATE_PX = 768
MSE_SEARCH_LIMIT = 50

# snippet for a parser script (json-to-csv) for the piezo cal data
"""
import json
with open("piezo_cal.json", "r") as f:
    foo = json.loads(f.read())
    for (step, z, piezo) in foo:
       print(f"{step}, {z}, {piezo}")
"""