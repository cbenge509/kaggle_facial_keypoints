#%%
############################################################################
# IMPORTS
############################################################################

import pandas as pd
import numpy as np
from utils import data_loader, data_transformer
import argparse
import pickle
import os

#%%
############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations (parameters)
DATA_PATH = "C:/kaggle/kaggle_keypoints/data"
PICKLE_PATH = "C:/kaggle/kaggle_keypoints/pickle"

# Competition file names (constants)
TRAIN_DATA_FILE = "training.csv"
TEST_DATA_FILE = "test.csv"
ID_FILE = "IdLookupTable.csv"
SAMPLE_SUBMISSION_FILE = "SampleSubmission.csv"
#PICKLE_FILE_TRAIN_AND_AUGMENT = "cleandata_train_plus_augmentation.pkl"
PICKLE_FILE_TRAIN30_AND_AUGMENT = "cleandata_train30_plus_augmentation.pkl"
PICKLE_FILE_TRAIN8_AND_AUGMENT = "cleandata_train8_plus_augmentation.pkl"

# Processing behavior (parameters)
VERBOSE = True
LOAD_FIXED_LABELS = True
#DROP_MISSING = False

# Augmentation behavior (parameters)
AUGMENTATION_ENABLE = True
AUGMENTATION_HORIZONTAL_FLIP = False
AUGMENTATION_ROTATION = True
AUGMENTATION_ROTATION_ANGLES = [9]
AUGMENTATION_BRIGHT_AND_DIM = True
AUGMENTATION_BRIGHT_LEVEL = [1.2]
AUGMENTATION_DIM_LEVEL = [0.6]
AUGMENTATION_PIXEL_SHIFTING = False
AUGMENTATION_PIXEL_SHIFTS = [12]
AUGMENTATION_ADD_NOISE = True
AUGMENTATION_STRETCH = True
AUGMENTATION_STRETCH_ALPHA = [991]
AUGMENTATION_STRETCH_SIGMA = [8]
AUGMENTATION_CONTRAST_STRETCHING = False
AUGMENTATION_CONTRAST_STRETCHING_ALPHA = [0.0]
AUGMENTATION_CONTRAST_STRETCHING_BETA = [1.2]
AUGMENTATION_SHARPENING = False
AUGMENTATION_SHARPENING_ROUNDS = [1]
AUGMENTATION_BLUR = False
AUGMENTATION_BLUR_ROUNDS = [1]
AUGMENTATION_SMOOTH = False
AUGMENTATION_SMOOTH_ROUNDS = [1]
AUGMENTATION_ADAPTIVE_HISTOGRAM = False
AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT = [0.03]

#%%

############################################################################
# ARGUMENT SPECIFICATION
############################################################################

parser = argparse.ArgumentParser(description = "Performs data preparation for the Kaggle Facial Keypoints Detection challenge.")
# Top-level arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-a', '--augment', action = 'store_true', help = 'Create an augmentation dataset and include it with the final train output.')
parser.add_argument('-f', '--fixed_labels', action = 'store_true', help = 'Enhance the training set with manually fixed training labels.')
parser.add_argument('-dp', '--data_path', type = str, default = "C:/kaggle/kaggle_keypoints/data", help = "Path to location of raw competition data files.")
parser.add_argument('-pp', '--pickle_path', type = str, default = "C:/kaggle/kaggle_keypoints/pickle", help = "Path to location of output pickle files (post processing files).")
#parser.add_argument('-d', '--drop_missing', action = 'store_true', help = 'Enables dropping training rows with missing labels (caution - drastically impacts distribution and size of train)')

# Augmentation 2nd-level, sub-arguments
parser.add_argument('-ahf', '--augment_horizontal_flip', action = 'store_true', help = 'Augment with horizontal flipping of images that contain all 15 keypoints in the training dataset..')
parser.add_argument('-ar', '--augment_rotation', action = 'store_true', help = 'Augment with +/- rotations of images that contain all 15 keypoints in the training dataset..')
parser.add_argument('-abd', '--augment_bright_and_dim', action = 'store_true', help = 'Augment with bright and dimming of images that contain all 15 keypoints in the training dataset.')
parser.add_argument('-aps', '--augment_pixel_shift', action = 'store_true', help = 'Augment with pixel shifting of images that contain all 15 keypoints in the training dataset.')
parser.add_argument('-acs', '--augment_contrast_stretch', action = 'store_true', help = 'Augment with contrast stretching of images that contain all 15 keypoints in the training dataset.')
parser.add_argument('-as', '--augment_elastic_stretch', action = 'store_true', help = 'Augment with elastic stretching of images that contain all 15 keypoints in the training dataset.')
parser.add_argument('-an', '--augment_noise', action = 'store_true', help = 'Augment with added noise in images that contain all 15 keypoints in the training dataset.')
parser.add_argument('-ash', '--augment_sharpen', action = 'store_true', help = 'Augment with image sharpening via simple kernel convolution.')
parser.add_argument('-asb', '--augment_blur', action = 'store_true', help = 'Augment with image blurring via simple kernel convolution.')
parser.add_argument('-ass', '--augment_smooth', action = 'store_true', help = 'Augment with image smoothing (Laplacian) via simple kernel convolution.')
parser.add_argument('-ah', '--augment_adaptive_histogram', action = 'store_true', help = 'Augment with Contrast Limited Adaptive Histogram Equalization (CLAHE).')

# Augmentation 3rd level
parser.add_argument('-ar_d', '--augment_rotation_degrees', nargs = '+', type = int, default = [9], help = 'Specifies the +/- rotation degrees for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-abd_b', '--augment_brightness', nargs = '+', type = float, default = [1.2], help = 'Specifies the brightness levels for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-abd_d', '--augment_dimness', nargs = '+', type = float, default = [0.6], help = 'Specifies the dimness levels for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-aps_d', '--augment_pixel_shift_degrees', nargs = '+', type = int, default = [9], help = 'Specifies the number of pixels for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-acs_a', '--augment_contrast_stretch_alpha', nargs = '+', type = float, default = [0.0], help = 'Specifies the alpha (lower) levels for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-acs_b', '--augment_contrast_stretch_beta', nargs = '+', type = float, default = [1.2], help = 'Specifies the beta (upper) levels for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-as_a', '--augment_elastic_stretch_alpha', nargs = '+', type = int, default = [991], help = 'Specifies the alpha for elastic stretching for the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-as_s', '--augment_elastic_stretch_sigma', nargs = '+', type = int, default = [8], help = 'Specifies the sigma (std) or the images that contain all 15 keypoints in the training dataset to be added as augmentation (list)')
parser.add_argument('-ash_r', '--augment_sharpen_rounds', nargs = '+', type = int, default = [1], help = 'Specifies the number of rounds to apply the sharpen convolution kernel to each image (list)')
parser.add_argument('-asb_r', '--augment_blur_rounds', nargs = '+', type = int, default = [1], help = 'Specifies the number of rounds to apply the blurring convolution kernel to each image (list)')
parser.add_argument('-ass_r', '--augment_smooth_rounds', nargs = '+', type = int, default = [1], help = 'Specifies the number of rounds to apply the smoothing convolution kernel to each image (list)')
parser.add_argument('-ah_c', '--augment_adaptive_histogram_clip_limit', nargs = '+', type = float, default = [0.03], help = 'Specifies the clip limit used for CLAHE augmentation.')

#%%

############################################################################
# ARGUMENT PARSING
############################################################################


def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, LOAD_FIXED_LABELS, AUGMENTATION_ENABLE, AUGMENTATION_HORIZONTAL_FLIP, DATA_PATH, PICKLE_PATH, \
        AUGMENTATION_ROTATION, AUGMENTATION_ROTATION_ANGLES, AUGMENTATION_BRIGHT_AND_DIM, AUGMENTATION_BRIGHT_LEVEL, \
        AUGMENTATION_DIM_LEVEL, AUGMENTATION_PIXEL_SHIFTING, AUGMENTATION_PIXEL_SHIFTS, AUGMENTATION_ADD_NOISE, \
        AUGMENTATION_STRETCH, AUGMENTATION_STRETCH_ALPHA, AUGMENTATION_STRETCH_SIGMA, AUGMENTATION_CONTRAST_STRETCHING, \
        AUGMENTATION_CONTRAST_STRETCHING_ALPHA, AUGMENTATION_CONTRAST_STRETCHING_BETA, AUGMENTATION_SHARPENING, \
        AUGMENTATION_SHARPENING_ROUNDS, AUGMENTATION_BLUR, AUGMENTATION_BLUR_ROUNDS, AUGMENTATION_SMOOTH, AUGMENTATION_SMOOTH_ROUNDS, \
        AUGMENTATION_ADAPTIVE_HISTOGRAM, AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT #, DROP_MISSING
    
    args = vars(parser.parse_args())

    if display_args:
        print("".join(["\nPREPARE_DATA Arguments in use:\n", "-" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Process 1st-level arguments
    VERBOSE = not args['no_verbose']
    LOAD_FIXED_LABELS = args['fixed_labels']
    AUGMENTATION_ENABLE = args['augment']
    #DROP_MISSING = args['drop_missing']

    DATA_PATH = str(args['data_path']).lower().strip().replace('\\', '/')
    PICKLE_PATH = str(args['pickle_path']).lower().strip().replace('\\', '/')

    # validate the presence of the paths
    for p, v, l in zip([DATA_PATH, PICKLE_PATH], ['data_path', 'pickle_path'], ['Competition data file path', 'Pickle file path (post-processing)']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    # process 2nd-level augmentation arguments
    if AUGMENTATION_ENABLE:
        
        AUGMENTATION_HORIZONTAL_FLIP = args['augment_horizontal_flip']
        AUGMENTATION_ROTATION = args['augment_rotation']
        AUGMENTATION_BRIGHT_AND_DIM = args['augment_bright_and_dim']
        AUGMENTATION_PIXEL_SHIFTING = args['augment_pixel_shift']
        AUGMENTATION_CONTRAST_STRETCHING = args['augment_contrast_stretch']
        AUGMENTATION_STRETCH = args['augment_elastic_stretch']
        AUGMENTATION_ADD_NOISE = args['augment_noise']
        AUGMENTATION_SHARPENING = args['augment_sharpen']
        AUGMENTATION_BLUR = args['augment_blur']
        AUGMENTATION_SMOOTH = args['augment_smooth']
        AUGMENTATION_ADAPTIVE_HISTOGRAM = args['augment_adaptive_histogram']

        # process 3rd-level augmentation arguments

        # Adaptive histogram
        if AUGMENTATION_ADAPTIVE_HISTOGRAM:
            AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT = args['augment_adaptive_histogram_clip_limit']
            # validate the user input
            assert type(AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT) is list, "Parameter `augment_adaptive_histogram_clip_limit` must be of type list."
            assert len(AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT) > 0, "Parameter `augment_adaptive_histogram_clip_limit` must contain at least one value."
            assert all(isinstance(x, float) for x in AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT), "All values in `augment_adaptive_histogram_clip_limit` list must be of type Float."
            assert all(0.0 < x <= 1.0 for x in AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT), "All values in `augment_adaptive_histogram_clip_limit` must be between 0.0 and 1.0."

        # Blur parameters
        if AUGMENTATION_BLUR:
            AUGMENTATION_BLUR_ROUNDS = args['augment_blur_rounds']
            # validate the user input
            assert type(AUGMENTATION_BLUR_ROUNDS) is list, "Parameter `augment_blur_rounds` must be of type list."
            assert len(AUGMENTATION_BLUR_ROUNDS) > 0, "Parameter `augment_blur_rounds` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_BLUR_ROUNDS), "All values in `augment_blur_rounds` list must be of type Int."
            assert all(0 < x < 6 for x in AUGMENTATION_BLUR_ROUNDS), "All values in `augment_blur_rounds` must be between 1 and 5."

        # Smooth parameters
        if AUGMENTATION_SMOOTH:
            AUGMENTATION_SMOOTH_ROUNDS = args['augment_smooth_rounds']
            # validate the user input
            assert type(AUGMENTATION_SMOOTH_ROUNDS) is list, "Parameter `augment_smooth_rounds` must be of type list."
            assert len(AUGMENTATION_SMOOTH_ROUNDS) > 0, "Parameter `augment_smooth_rounds` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_SMOOTH_ROUNDS), "All values in `augment_smooth_rounds` list must be of type Int."
            assert all(0 < x < 6 for x in AUGMENTATION_SMOOTH_ROUNDS), "All values in `augment_smooth_rounds` must be between 1 and 5."

        # Sharpening parameters
        if AUGMENTATION_SHARPENING:
            AUGMENTATION_SHARPENING_ROUNDS = args['augment_sharpen_rounds']
            # validate the user input
            assert type(AUGMENTATION_SHARPENING_ROUNDS) is list, "Parameter `augment_sharpen_rounds` must be of type list."
            assert len(AUGMENTATION_SHARPENING_ROUNDS) > 0, "Parameter `augment_sharpen_rounds` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_SHARPENING_ROUNDS), "All values in `augment_sharpen_rounds` list must be of type Int."
            assert all(0 < x < 6 for x in AUGMENTATION_SHARPENING_ROUNDS), "All values in `augment_sharpen_rounds` must be between 1 and 5."

        # Rotation parameters
        if AUGMENTATION_ROTATION:
            AUGMENTATION_ROTATION_ANGLES = args['augment_rotation_degrees']
            # validate the user input
            assert type(AUGMENTATION_ROTATION_ANGLES) is list, "Parameter `augment_rotation_degrees` must be of type list."
            assert len(AUGMENTATION_ROTATION_ANGLES) > 0, "Parameter `augment_rotation_degrees` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_ROTATION_ANGLES), "All values in `augment_rotation_degrees` list must be of type Int."
            assert all(0 < x < 181 for x in AUGMENTATION_ROTATION_ANGLES), "All values in `augment_rotation_degrees` must be between 1 and 180."

            # ensure no duplicates are present
            AUGMENTATION_ROTATION_ANGLES = list(np.unique(AUGMENTATION_ROTATION_ANGLES))

        # Bright and dim augmentation parameters
        if AUGMENTATION_BRIGHT_AND_DIM:
            AUGMENTATION_BRIGHT_LEVEL = args['augment_brightness']
            # validate the user input
            assert type(AUGMENTATION_BRIGHT_LEVEL) is list, "Parameter `augment_brightness` must be of type list."
            assert len(AUGMENTATION_BRIGHT_LEVEL) > 0, "Parameter `augment_brightness` must contain at least one value."
            assert all(isinstance(x, float) for x in AUGMENTATION_BRIGHT_LEVEL), "All values in `augment_brightness` list must be of type Float."
            assert all(1.0 < x < 5.0 for x in AUGMENTATION_BRIGHT_LEVEL), "All values in `augment_brightness` must be between 1.1 and 4.99."

            AUGMENTATION_DIM_LEVEL = args['augment_dimness']
            # validate the user input
            assert type(AUGMENTATION_DIM_LEVEL) is list, "Parameter `augment_dimness` must be of type list."
            assert len(AUGMENTATION_DIM_LEVEL) > 0, "Parameter `augment_dimness` must contain at least one value."
            assert all(isinstance(x, float) for x in AUGMENTATION_DIM_LEVEL), "All values in `augment_dimness` list must be of type Float."
            assert all(0.09 < x < 1.0 for x in AUGMENTATION_DIM_LEVEL), "All values in `augment_dimness` must be between 0.1 and 0.99."
            
            assert len(AUGMENTATION_BRIGHT_LEVEL) == len(AUGMENTATION_DIM_LEVEL), "Parameter `augment_brightness` and `augment_dimness` must be of the same length."

            # ensure no duplicates are present
            combined = list(zip(AUGMENTATION_BRIGHT_LEVEL, AUGMENTATION_DIM_LEVEL))
            combined = list(dict.fromkeys(combined))
            AUGMENTATION_BRIGHT_LEVEL, AUGMENTATION_DIM_LEVEL = [], []
            for b, d in combined:
                AUGMENTATION_BRIGHT_LEVEL.append(b)
                AUGMENTATION_DIM_LEVEL.append(d)

        # Pixel-shifting augmentation parameters
        if AUGMENTATION_PIXEL_SHIFTING:
            AUGMENTATION_PIXEL_SHIFTS = args['augment_pixel_shift_degrees']
            # validate the user input
            assert type(AUGMENTATION_PIXEL_SHIFTS) is list, "Parameter `augment_pixel_shift_degrees` must be of type list."
            assert len(AUGMENTATION_PIXEL_SHIFTS) > 0, "Parameter `augment_pixel_shift_degrees` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_PIXEL_SHIFTS), "All values in `augment_pixel_shift_degrees` list must be of type Float."
            assert all(0 < x < 51 for x in AUGMENTATION_PIXEL_SHIFTS), "All values in `augment_pixel_shift_degrees` must be between 1 and 50."

            # ensure no duplicates are present
            AUGMENTATION_PIXEL_SHIFTS = list(np.unique(AUGMENTATION_PIXEL_SHIFTS))

        # Elastic image stretch augmentation parameters
        if AUGMENTATION_STRETCH:
            AUGMENTATION_STRETCH_ALPHA = args['augment_elastic_stretch_alpha']
            # validate the user input
            assert type(AUGMENTATION_STRETCH_ALPHA) is list, "Parameter `augment_elastic_stretch_alpha` must be of type list."
            assert len(AUGMENTATION_STRETCH_ALPHA) > 0, "Parameter `augment_elastic_stretch_alpha` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_STRETCH_ALPHA), "All values in `augment_elastic_stretch_alpha` list must be of type int."
            assert all(499 < x < 2001 for x in AUGMENTATION_STRETCH_ALPHA), "All values in `augment_elastic_stretch_alpha` must be between 500 and 2000."

            AUGMENTATION_STRETCH_SIGMA = args['augment_elastic_stretch_sigma']
            # validate the user input
            assert type(AUGMENTATION_STRETCH_SIGMA) is list, "Parameter `augment_elastic_stretch_sigma` must be of type list."
            assert len(AUGMENTATION_STRETCH_SIGMA) > 0, "Parameter `augment_elastic_stretch_sigma` must contain at least one value."
            assert all(isinstance(x, int) for x in AUGMENTATION_STRETCH_SIGMA), "All values in `augment_elastic_stretch_sigma` list must be of type int."
            assert all(0 < x < 26 for x in AUGMENTATION_STRETCH_SIGMA), "All values in `augment_elastic_stretch_sigma` must be between 1 and 25."

            assert len(AUGMENTATION_STRETCH_ALPHA) == len(AUGMENTATION_STRETCH_SIGMA), "Parameter `augment_elastic_stretch_alpha` and `augment_elastic_stretch_sigma` must be of the same length."

            # ensure no duplicates are present
            combined = list(zip(AUGMENTATION_STRETCH_ALPHA, AUGMENTATION_STRETCH_SIGMA))
            combined = list(dict.fromkeys(combined))
            AUGMENTATION_STRETCH_ALPHA, AUGMENTATION_STRETCH_SIGMA = [], []
            for a, s in combined:
                AUGMENTATION_STRETCH_ALPHA.append(a)
                AUGMENTATION_STRETCH_SIGMA.append(s)

        # Constrast stretching parameters
        if AUGMENTATION_CONTRAST_STRETCHING:
            AUGMENTATION_CONTRAST_STRETCHING_ALPHA = args['augment_contrast_stretch_alpha']
            # validate the user input
            assert type(AUGMENTATION_CONTRAST_STRETCHING_ALPHA) is list, "Parameter `augment_contrast_stretch_alpha` must be of type list."
            assert len(AUGMENTATION_CONTRAST_STRETCHING_ALPHA) > 0, "Parameter `augment_contrast_stretch_alpha` must contain at least one value."
            assert all(isinstance(x, float) for x in AUGMENTATION_CONTRAST_STRETCHING_ALPHA), "All values in `augment_contrast_stretch_alpha` list must be of type float."
            assert all(-0.01 < x < 0.16 for x in AUGMENTATION_CONTRAST_STRETCHING_ALPHA), "All values in `augment_contrast_stretch_alpha` must be between 0.0 and 0.15 (recommended: 0.0)."

            AUGMENTATION_CONTRAST_STRETCHING_BETA = args['augment_contrast_stretch_beta']
            # validate the user input
            assert type(AUGMENTATION_CONTRAST_STRETCHING_BETA) is list, "Parameter `augment_contrast_stretch_beta` must be of type list."
            assert len(AUGMENTATION_CONTRAST_STRETCHING_BETA) > 0, "Parameter `augment_contrast_stretch_beta` must contain at least one value."
            assert all(isinstance(x, float) for x in AUGMENTATION_CONTRAST_STRETCHING_BETA), "All values in `augment_contrast_stretch_beta` list must be of type float."
            assert all(1.0 < x < 3.09 for x in AUGMENTATION_CONTRAST_STRETCHING_BETA), "All values in `augment_contrast_stretch_beta` must be between 1.1 and 4.0."

            assert len(AUGMENTATION_CONTRAST_STRETCHING_ALPHA) == len(AUGMENTATION_CONTRAST_STRETCHING_BETA), "Parameter `augment_contrast_stretch_alpha` and `augment_contrast_stretch_beta` must be of the same length."

            # ensure no duplicates are present
            combined = list(zip(AUGMENTATION_CONTRAST_STRETCHING_ALPHA, AUGMENTATION_CONTRAST_STRETCHING_BETA))
            combined = list(dict.fromkeys(combined))
            AUGMENTATION_CONTRAST_STRETCHING_ALPHA, AUGMENTATION_CONTRAST_STRETCHING_BETA = [], []
            for a, b in combined:
                AUGMENTATION_CONTRAST_STRETCHING_ALPHA.append(a)
                AUGMENTATION_CONTRAST_STRETCHING_BETA.append(b)


#%%

############################################################################
# LOAD RAW DATA
############################################################################

# Load the raw data from the CSV files; performs only data-type casting and no other modification
def load_raw_data(data_path, pickle_path, train_data_file, test_data_file, id_lookup_file,
        sample_submission_file, verbose = True):
    
    # Instanciate the DataLoader helper class
    dl = data_loader.DataLoader(data_path = data_path, pickle_path = pickle_path, 
        train_data_file = train_data_file, test_data_file = test_data_file, 
        id_lookup_file = id_lookup_file, sample_submission_file = sample_submission_file, 
        verbose = VERBOSE)

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN RAW DATA LOAD <<<\n", "-" * 50, "\n"]))
    # Retrive the raw test and raw train dataframes
    ids, _, raw_test, raw_train = dl.LoadRawData(verbose = verbose, recalculate_pickle = True)
    if verbose: print("".join(["\n", "-" * 50, "\n>>> END RAW DATA LOAD <<<\n", "-" * 50, "\n"]))

    return raw_train, raw_test, ids

#%%

############################################################################
# PIPELINE - DATA CLEANING
############################################################################

# Perform baseline clean-up on the data
def clean_data(raw_train, raw_test, ids, fix_labels = True, drop_missing = False, verbose = True):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN DATA CLEANING <<<\n", "-" * 50, "\n"]))
    _, _, _, train30, train8, _, _ = xform.CleanData(train = raw_train, test = raw_test, verbose = verbose, 
        recalculate_pickle = True, fix_labels = fix_labels, drop_missing = drop_missing, ids = ids)
    if verbose: print("".join(["\n", "-" * 50, "\n>>> END DATA CLEANING <<<\n", "-" * 50, "\n"]))

    return train30, train8

# %%
############################################################################
# PIPELINE - DATA AUGMENTATION
############################################################################

def augment_data(enable, train, horizontal_flip, rotation, rotation_angles, bright_and_dim, bright_level,
        dim_level, shifting, pixel_shifts, add_noise, stretch, stretch_alpha, stretch_sigma,
        contrast_stretch, contrast_alpha, contrast_beta, sharpen, sharpen_rounds, blur, blur_rounds, 
        smooth, smooth_rounds, adaptive_histogram, adaptive_histogram_clip_limit, verbose = True, full = True):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN DATA AUGMENTATION <<<\n", "-" * 50, "\n"]))
    if enable:
        # get an augmented data set to combine with TRAIN
        augmented = xform.AugmentData(train, horizontal_flip = horizontal_flip, 
            rotation = rotation, rotation_angles = rotation_angles,
            bright_and_dim = bright_and_dim, bright_level = bright_level,
            dim_level = dim_level, shifting = shifting,
            pixel_shifts = pixel_shifts, add_noise = add_noise, verbose = verbose, 
            recalculate_pickle = True, stretch = stretch, stretch_alpha = stretch_alpha, 
            stretch_sigma = stretch_sigma, contrast_stretch = contrast_stretch, 
            contrast_alpha = contrast_alpha, contrast_beta = contrast_beta, sharpen = sharpen,
            sharpen_rounds = sharpen_rounds, blur = blur, blur_rounds = blur_rounds,
            smooth = smooth, smooth_rounds = smooth_rounds, adaptive_histogram = adaptive_histogram,
            adaptive_histogram_clip_limit = adaptive_histogram_clip_limit, full = full)
    else:
        augmented = train[(train.index == -1)]
        print("--- Augmentation Skipped ---")

    if verbose: print("".join(["-" * 50, "\n>>> END DATA AUGMENTATION <<<\n", "-" * 50, "\n"]))

    return augmented

#%%

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == "__main__":

    # Clear the screen
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    
    # Process command-line arguments and set parameters
    process_arguments(parser.parse_args(), display_args = True)

    # Load the raw data
    raw_train, raw_test, ids = load_raw_data(data_path = DATA_PATH, pickle_path = PICKLE_PATH, 
        train_data_file = TRAIN_DATA_FILE, test_data_file = TEST_DATA_FILE, id_lookup_file = ID_FILE,
        sample_submission_file = SAMPLE_SUBMISSION_FILE, verbose = VERBOSE)

    # Instanciate the XForm data transformer helper class
    xform = data_transformer.Xform(pickle_path = PICKLE_PATH, verbose = VERBOSE)

    # Clean the data
    train30, train8 = clean_data(raw_train, raw_test, ids = ids, fix_labels = LOAD_FIXED_LABELS, verbose = VERBOSE) #, drop_missing = DROP_MISSING)

    # Augment the data
    augmented30 = augment_data(enable = AUGMENTATION_ENABLE, train = train30, horizontal_flip = AUGMENTATION_HORIZONTAL_FLIP, 
            rotation = AUGMENTATION_ROTATION, rotation_angles = AUGMENTATION_ROTATION_ANGLES,
            bright_and_dim = AUGMENTATION_BRIGHT_AND_DIM, bright_level = AUGMENTATION_BRIGHT_LEVEL,
            dim_level = AUGMENTATION_DIM_LEVEL, shifting = AUGMENTATION_PIXEL_SHIFTING,
            pixel_shifts = AUGMENTATION_PIXEL_SHIFTS, add_noise = AUGMENTATION_ADD_NOISE, verbose = VERBOSE, 
            stretch = AUGMENTATION_STRETCH, stretch_alpha = AUGMENTATION_STRETCH_ALPHA, 
            stretch_sigma = AUGMENTATION_STRETCH_SIGMA, contrast_stretch = AUGMENTATION_CONTRAST_STRETCHING, 
            contrast_alpha = AUGMENTATION_CONTRAST_STRETCHING_ALPHA, contrast_beta = AUGMENTATION_CONTRAST_STRETCHING_BETA,
            sharpen = AUGMENTATION_SHARPENING, sharpen_rounds = AUGMENTATION_SHARPENING_ROUNDS,
            blur = AUGMENTATION_BLUR, blur_rounds = AUGMENTATION_BLUR_ROUNDS, smooth = AUGMENTATION_SMOOTH,
            smooth_rounds = AUGMENTATION_SMOOTH_ROUNDS, adaptive_histogram = AUGMENTATION_ADAPTIVE_HISTOGRAM,
            adaptive_histogram_clip_limit = AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT, full = True)

    augmented8 = augment_data(enable = AUGMENTATION_ENABLE, train = train8, horizontal_flip = AUGMENTATION_HORIZONTAL_FLIP, 
            rotation = AUGMENTATION_ROTATION, rotation_angles = AUGMENTATION_ROTATION_ANGLES,
            bright_and_dim = AUGMENTATION_BRIGHT_AND_DIM, bright_level = AUGMENTATION_BRIGHT_LEVEL,
            dim_level = AUGMENTATION_DIM_LEVEL, shifting = AUGMENTATION_PIXEL_SHIFTING,
            pixel_shifts = AUGMENTATION_PIXEL_SHIFTS, add_noise = AUGMENTATION_ADD_NOISE, verbose = VERBOSE, 
            stretch = AUGMENTATION_STRETCH, stretch_alpha = AUGMENTATION_STRETCH_ALPHA, 
            stretch_sigma = AUGMENTATION_STRETCH_SIGMA, contrast_stretch = AUGMENTATION_CONTRAST_STRETCHING, 
            contrast_alpha = AUGMENTATION_CONTRAST_STRETCHING_ALPHA, contrast_beta = AUGMENTATION_CONTRAST_STRETCHING_BETA,
            sharpen = AUGMENTATION_SHARPENING, sharpen_rounds = AUGMENTATION_SHARPENING_ROUNDS,
            blur = AUGMENTATION_BLUR, blur_rounds = AUGMENTATION_BLUR_ROUNDS, smooth = AUGMENTATION_SMOOTH,
            smooth_rounds = AUGMENTATION_SMOOTH_ROUNDS, adaptive_histogram = AUGMENTATION_ADAPTIVE_HISTOGRAM,
            adaptive_histogram_clip_limit = AUGMENTATION_ADAPTIVE_HISTOGRAM_CLIP_LIMIT, full = False)

    # combine the train + augmentation dataframes and save as a pickle'd dataframe
    if PICKLE_PATH.endswith('/'): sep_add = ""
    else: sep_add = "/"
    PICKLE_FILE_TRAIN30_AND_AUGMENT = "".join([PICKLE_PATH, sep_add, PICKLE_FILE_TRAIN30_AND_AUGMENT])
    PICKLE_FILE_TRAIN8_AND_AUGMENT = "".join([PICKLE_PATH, sep_add, PICKLE_FILE_TRAIN8_AND_AUGMENT])
    
    if augmented30.shape[0] > 0:
        train30 = train30.append(augmented30, ignore_index = True).reset_index().drop(columns = ['index'])
    if augmented8.shape[0] > 0:
        train8 = train8.append(augmented8, ignore_index = True).reset_index().drop(columns = ['index'])
    
    # note: if augmentation isn't called, this file will be a duplicate of the cleaned train dataset
    if VERBOSE: print("Writing TRAIN30 + AUGMENTATION pickle file to '%s'..." % PICKLE_FILE_TRAIN30_AND_AUGMENT)
    pickle.dump(train30, open(PICKLE_FILE_TRAIN30_AND_AUGMENT, "wb"))
    if VERBOSE: print("Writing TRAIN8 + AUGMENTATION pickle file to '%s'..." % PICKLE_FILE_TRAIN8_AND_AUGMENT)
    pickle.dump(train8, open(PICKLE_FILE_TRAIN8_AND_AUGMENT, "wb"))
    
    # signal the end of data preparation
    if VERBOSE: print("".join(["-" * 50, "\n>>> DATA PREPARATION COMPLETE <<<\n", "-" * 50, "\n"]))
