X_RES = 3840
Y_RES = 2160
PIX_PER_UM_20X = 3535 / 370 # 20x objective
PIX_PER_UM_5X = 2350 / 1000 # 5x objective, 3.94 +/- 0.005 ratio to 20x
PIX_PER_UM_10X = 3330 / 700 # 10x objective, ~4.757 pix/um
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

INITIAL_R = 1

TILES_VERSION = 1

PIEZO_UM_PER_LSB= 0.0058812 # from empirical measurements
SECULAR_PIEZO_UM_PER_LSB = 0.007425