#%%
############################################################################
# IMPORTS
############################################################################

import pandas as pd
import numpy as np
from utils import model_zoo, data_transformer

import argparse
import pickle
import os


#%%
############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations and model name (parameters)
MODEL_NAME = "KERAS_LENET5"
PICKLE_PATH = "C:/kaggle/kaggle_keypoints/pickle"
MODEL_PATH = "C:/kaggle/kaggle_keypoints/models"

# Processing behavior (parameters)
NORMALIZE_LABELS = False
VERBOSE = True
BATCH_SIZE = 128
EPOCH_COUNT = 300
USE_AUGMENTATION = True
USE_VALIDATION = False
VALIDATION_SPLIT = 0.1
USE30 = True
TRAIN_DATA_AUGMENTATION_FILE = "cleandata_train_plus_augmentation.pkl"
TRAIN_DATA_FILE = "cleandata_naive_train.pkl"

# Processing behavior (constants)
AVAILABLE_MODELS = ["KERAS_LENET5", "KERAS_INCEPTION", "KERAS_KAGGLE1", "KERAS_NAIMISHNET", "KERAS_CONVNET5", "KERAS_INCEPTIONV3", "KERAS_KAGGLE2", "KERAS_RESNET50", "KERAS_RESNET", "KERAS_RESNEXT50"]
VALIDATION_DATA_FILE = "validation_set.pkl"

TRAIN30_DATA_AUGMENTATION_FILE = "cleandata_train30_plus_augmentation.pkl"
TRAIN8_DATA_AUGMENTATION_FILE = "cleandata_train8_plus_augmentation.pkl"

TRAIN30_DATA_FILE = "cleandata_train30.pkl"
TRAIN8_DATA_FILE = "cleandata_train8.pkl"

#%%

############################################################################
# ARGUMENT SPECIFICATION
############################################################################

parser = argparse.ArgumentParser(description = "Performs model training for the Kaggle Facial Keypoints Detection challenge.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-pa', '--partial', action = 'store_true', help = 'Trains using the partial (8-value) dataset instead of the full (30-value) dataset.')
parser.add_argument('-pp', '--pickle_path', type = str, default = "C:/kaggle/kaggle_keypoints/pickle", help = "Path to location of output pickle files (post processing files).")
parser.add_argument('-mp', '--model_path', type = str, default = "C:/kaggle/kaggle_keypoints/models", help = "Path to location of output model files.")
parser.add_argument('-m', '--model_name', type = str, default = "KERAS_LENET5", help = "Name of the model to train.")
parser.add_argument('-nl', '--normalize_labels', action = 'store_true', help = "Enables the normalization of prediction label values prior to training.")
parser.add_argument('-bs', '--batch_size', type = int, default = 128, help = "Specifies the batch size for neural network training.")
parser.add_argument('-e', '--epochs', type = int, default = 300, help = "Specifies the total epochs for neural network training.")
parser.add_argument('-na', '--no_augmentation', action = 'store_true', help = 'Disables the use of augmentation data in train.')
parser.add_argument('-v', '--validation', action = 'store_true', help = 'Enable the use of a validation set (disables the use of a validation_split).')
parser.add_argument('-vs', '--validation_split', type = float, default = 0.1, help = 'Specifies the validation split percentage (if not using a validation set).')

############################################################################
# ARGUMENT PARSING
############################################################################


def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, PICKLE_PATH, MODEL_PATH, MODEL_NAME, NORMALIZE_LABELS, BATCH_SIZE, EPOCH_COUNT, \
           USE_AUGMENTATION, VALIDATION_SPLIT, USE_VALIDATION, USE30
    
    args = vars(parser.parse_args())

    if display_args:
        print("".join(["\TRAIN_MODEL Arguments in use:\n", "-" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    USE30 = not args['partial']
    USE_AUGMENTATION = not args['no_augmentation']
    MODEL_NAME = args['model_name']
    NORMALIZE_LABELS = args['normalize_labels']
    BATCH_SIZE = args['batch_size']
    EPOCH_COUNT = args['epochs']
    USE_VALIDATION = args['validation']
    VALIDATION_SPLIT = args['validation_split']
    MODEL_PATH = str(args['model_path']).lower().strip().replace('\\', '/')
    PICKLE_PATH = str(args['pickle_path']).lower().strip().replace('\\', '/')

    # validate the presence of the paths
    for p, v, l in zip([MODEL_PATH, PICKLE_PATH], ['model_path', 'pickle_path'], ['Model file path', 'Pickle file path']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    # validate the parameters entered
    assert 0 < BATCH_SIZE < 2049, "Parameter `batch_size` must be between 1 and 2,048."
    assert 0 < EPOCH_COUNT < 1000, "Parameter `epochs` must be between 1 and 1,000."
    if not USE_VALIDATION:
        assert 0.04 < VALIDATION_SPLIT < 0.51, "Parameter `validation_split` must be between 0.05 and 0.5."
    if not MODEL_NAME in AVAILABLE_MODELS:
        raise RuntimeError("Parameter `model_name` value of '%s' is invalid.  Must be in list: %s" % (MODEL_NAME, str(AVAILABLE_MODELS)))


#%%

############################################################################
# LOAD DATA
############################################################################

# load the data for training
def load_data(pickle_path, train_file, validation_file, use_validation, verbose = True):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN LOAD DATA <<<\n", "-" * 50, "\n"]))
    if not pickle_path.endswith("/"): pickle_path = "".join([pickle_path, "/"])
    train_file = "".join([pickle_path, train_file])

    if not os.path.isfile(train_file):
        raise RuntimeError("Train file '%s' not found - training cancelled." % train_file)

    train = pickle.load(open(train_file, "rb"))
    if verbose: print("Train file '%s' loaded; shape: %s" % (train_file, str(train.shape)))

    if use_validation:
        validation_file = "".join([pickle_path, validation_file])

        if not os.path.isfile(validation_file):
            raise RuntimeError("Validation file '%s' not found - training cancelled." % validation_file)

        validation = pickle.load(open(validation_file, "rb"))
        if verbose: print("Validation file '%s' loaded; shape: %s -- %.2f%% of train size." % (validation_file, str(validation.shape), (validation.shape[0] / train.shape[0]) * 100.))
    else:
        if verbose: print("No validation file specified; using with validation split value %.2f%% instead." % (VALIDATION_SPLIT * 100.))
        validation = None

    if verbose: print("".join(["\n", "-" * 50, "\n>>> END LOAD DATA <<<\n", "-" * 50, "\n"]))

    return train, validation

# %%
############################################################################
# TRAIN MODEL (GENERIC HANDLER)
############################################################################

def train_model(model_path, pickle_path, model_name, batch_size, epochs, normalize_labels, train, validation, use_validation, 
        validation_split, model_performance_file, verbose = True, full = True, skip_history = False):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN TRAINING ON %s <<<\n" % model_name, "-" * 50, "\n"]))

    # load helper modules for models and data transformation
    models = model_zoo.Models(model_path = MODEL_PATH)
    xform = data_transformer.Xform(pickle_path = PICKLE_PATH, verbose = VERBOSE)

    # validate the existence of the model output path; if it doesn't exist, create it
    if model_path.endswith("/"): sep_add = ""
    else: sep_add = "/"
    validate_path = "".join([model_path, sep_add, model_name])
    if not os.path.exists(validate_path):
        if verbose: print("Model output path '%s' does not yet exist, creating it." % validate_path)
        os.makedirs(validate_path)

    models_out, features = [], []
    # call the training module specific to the algorithm called
    if model_name == "KERAS_LENET5":
        feature_name = "ALL_FEATURES"
        report_metrics, report_names = ['val_mse'], ['output']
        model, hist_params, hist = train_model_lenet5(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(model)
        features.append(feature_name)

    elif model_name == "KERAS_INCEPTION":
        feature_name = "ALL_FEATURES"
        report_metrics = ['val_main_output_mse','val_auxilliary_output_1_mse','val_auxilliary_output_2_mse']
        report_names = ['main_output','auxilliary_1_output', 'auxilliary_2_output']
        main, aux1, aux2, hist_params, hist = train_model_inception(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.extend([main, aux1, aux2])
        features.append(feature_name)

    elif model_name == "KERAS_INCEPTIONV3":
        feature_name = "ALL_FEATURES"
        report_metrics = ['val_mse']
        report_names = ['main_output']
        main, hist_params, hist = train_model_inceptionv3(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(main)
        features.append(feature_name)

    elif model_name == "KERAS_RESNET50":
        feature_name = "ALL_FEATURES"
        report_metrics = ['val_mse']
        report_names = ['main_output']
        main, hist_params, hist = train_model_resnet50(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(main)
        features.append(feature_name)

    elif model_name == "KERAS_RESNET":
        feature_name = "ALL_FEATURES"
        report_metrics = ['val_mse']
        report_names = ['main_output']
        main, hist_params, hist = train_model_resnet(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(main)
        features.append(feature_name)

    elif model_name == "KERAS_RESNEXT50":
        feature_name = "ALL_FEATURES"
        report_metrics = ['val_mse']
        report_names = ['main_output']
        main, hist_params, hist = train_model_resnext50(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(main)
        features.append(feature_name)

    elif model_name == "KERAS_KAGGLE1":
        feature_name = "ALL_FEATURES"
        report_metrics, report_names = ['val_mse'], ['output']
        model, hist_params, hist = train_model_kaggle1(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(model)
        features.append(feature_name)

    elif model_name == "KERAS_KAGGLE2":
        feature_name = "ALL_FEATURES"
        report_metrics, report_names = ['val_mse'], ['output']
        model, hist_params, hist = train_model_kaggle2(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(model)
        features.append(feature_name)

    elif model_name == "KERAS_CONVNET5":
        feature_name = "ALL_FEATURES"
        report_metrics, report_names = ['val_mse'], ['output']
        model, hist_params, hist = train_model_convnet5(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.append(model)
        features.append(feature_name)

    elif model_name == "KERAS_NAIMISHNET":
        if full:
            feature_name = ['left_eye_center', 'right_eye_center', 'left_eye_inner_corner', 'left_eye_outer_corner', 
                'right_eye_inner_corner', 'right_eye_outer_corner', 'left_eyebrow_inner_end', 'left_eyebrow_outer_end', 
                'right_eyebrow_inner_end', 'right_eyebrow_outer_end', 'nose_tip', 'mouth_left_corner', 'mouth_right_corner', 
                'mouth_center_top_lip', 'mouth_center_bottom_lip']
        else:
            feature_name = ['left_eye_center', 'right_eye_center', 'nose_tip', 'mouth_center_bottom_lip']
        report_metrics, report_names = ['val_mse'], ['output']
        models_naimishnet, hist_params, hist = train_model_naimishnet(models = models, xform = xform, batch_size = batch_size, epochs = epochs, 
            normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, feature_name = feature_name, full = full, verbose = verbose)
        models_out.extend(models_naimishnet)
        features.extend(feature_name)

    else:
        raise RuntimeError("Model name '%s' not understood; cancelling training." % model_name)

    if not skip_history:
        output_model_history(model_path = model_path, model_name = model_name, feature_name = feature_name, hist_params = hist_params, 
            hist = hist, report_metrics = report_metrics, report_names = report_names, model_performance_file = model_performance_file, full = full, verbose = verbose)

    if verbose: print("".join(["-" * 50, "\n>>> END TRAINING ON %s <<<\n" % model_name, "-" * 50, "\n"]))

    return models_out, features
# %%

############################################################################
# TRAIN MODEL LENET5
############################################################################

def train_model_lenet5(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_lenet5(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_lenet5(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL KAGGLE1
############################################################################

def train_model_kaggle1(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_kaggle1(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_kaggle1(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL KAGGLE2
############################################################################

def train_model_kaggle2(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_kaggle2(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_kaggle2(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL RESNET
############################################################################

def train_model_resnet(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_resnet(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_resnet(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL RESNET50
############################################################################

def train_model_resnet50(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_resnet50(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_resnet50(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL RESNET50
############################################################################

def train_model_resnext50(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_resnext50(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_resnext50(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL INCEPTIONV3
############################################################################

def train_model_inceptionv3(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_inceptionv3(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_inceptionv3(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL CONVNET5
############################################################################

def train_model_convnet5(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        model, hist_params, hist = models.get_keras_convnet5(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        model, hist_params, hist = models.get_keras_convnet5(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL INCEPTION
############################################################################

def train_model_inception(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    X, Y = xform.PrepareTrain(train = train, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

    if not use_validation:
        if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
        main_model, aux1_model, aux2_model, hist_params, hist = models.get_keras_inception(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            val_split = validation_split, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)
    else:
        if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
        X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = feature_name, normalize = normalize_labels, verbose = verbose)

        main_model, aux1_model, aux2_model, hist_params, hist = models.get_keras_inception(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
            X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = feature_name, recalculate_pickle = True, full = full, verbose = verbose)

    return main_model, aux1_model, aux2_model, hist_params, hist

#%%

############################################################################
# TRAIN MODEL NAIMISHNET
############################################################################

def train_model_naimishnet(models, xform, batch_size, epochs, normalize_labels, train, validation, use_validation, validation_split, feature_name, full = True, verbose = True):

    hp, h, models_out = {}, {}, []
    for f in feature_name:
        
        X, Y = xform.PrepareTrain(train = train, feature_name = f, normalize = normalize_labels, verbose = verbose)

        if not use_validation:
            if verbose: print("Training using a validation split of %.2f%%." % (validation_split * 100.))
            model, hist_params, hist = models.get_keras_naimishnet(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
                val_split = validation_split, shuffle = True, feature_name = f, recalculate_pickle = True, full = full, verbose = verbose)
        else:
            if verbose: print("Training using a validation dataset with shape: %s" % str(validation.shape))
            X_val, Y_val = xform.PrepareTrain(train = validation, feature_name = f, normalize = normalize_labels, verbose = verbose)

            model, hist_params, hist = models.get_keras_naimishnet(X = X, Y = Y, batch_size = batch_size, epoch_count = epochs,
                X_val = X_val, Y_val = Y_val, shuffle = True, feature_name = f, recalculate_pickle = True, full = full, verbose = verbose)

        models_out.append(model)
        hp[f] = hist_params
        h[f] = hist

    return models_out, hp, h

#%%

############################################################################
# OUTPUT TRAINING HISTORY
############################################################################

def output_model_history(model_path, model_name, feature_name, hist_params, hist, report_metrics, report_names, model_performance_file, full = True, verbose = True):

    if not type(feature_name) is list:
        feature_name = [feature_name]
    if not type(hist) is dict:
        hh = {}
        hh[feature_name[0]] = hist
        hist = hh
    if not type(hist_params) is dict:
        hp = {}
        hp[feature_name[0]] = hist_params
        hist_params = hp

    performance = {}
    d_build = {}
    for keypoint in feature_name:
        h = hist[keypoint]
        hp = hist_params[keypoint]
        for n, m in zip(report_names, report_metrics):
            if verbose:
                print("Training of model %s for feature '%s' completed. Best %s epoch: [%d], Best %s: [%.5f]." % 
                    (model_name, keypoint, n, np.argmin(h[m].values) + 1, m, h.iloc[np.argmin(h[m].values), h.columns.get_loc(m)]))

            d_build.update({
                "".join([n,"_epoch"]):np.argmin(h[m].values) + 1,
                m:h.iloc[np.argmin(h[m].values), h.columns.get_loc(m)]})
        
        performance[keypoint] = d_build

        hist_file = model_performance_file.replace("performance_", "".join(["parameters_", keypoint, "_"]))
        if verbose: print("Writing model parameters out to disk at '%s'" % hist_file)
        pd.DataFrame(hp).to_csv(hist_file, index = False)

    # save all baseline performance & parameter history
    if verbose: print("Writing summary training performance metrics out to disk at '%s'" % model_performance_file)
    pd.DataFrame(performance).T.to_csv(model_performance_file, index = False)
    
    return

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

    # load train and test
    if USE_AUGMENTATION: 
        if USE30: train_file = TRAIN30_DATA_AUGMENTATION_FILE
        else: train_file = TRAIN8_DATA_AUGMENTATION_FILE
    else: 
        if USE30: train_file = TRAIN30_DATA_FILE
        else: train_file = TRAIN8_DATA_FILE

    train, validation = load_data(pickle_path = PICKLE_PATH, train_file = train_file, 
        validation_file = VALIDATION_DATA_FILE, use_validation = USE_VALIDATION, verbose = VERBOSE)

    # derive the model performance file location
    output_path = str(MODEL_PATH).replace("\\", "/").strip()
    if not output_path.endswith('/'): output_path = "".join((output_path, "/", MODEL_NAME, "/"))
    if not os.path.exists(output_path):
        print("Creating path '%s'." % output_path)
        os.makedirs(output_path)
    if USE30:
        iter_num = max(([0] + [int(f.replace("all_performance_30_", "")[:-4]) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and f.startswith('all_performance_30_')])) + 1
        model_performance_file = "".join([output_path, "all_performance_30_", str(iter_num), ".csv"])
    else:
        iter_num = max(([0] + [int(f.replace("all_performance_8_", "")[:-4]) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and f.startswith('all_performance_8_')])) + 1
        model_performance_file = "".join([output_path, "all_performance_8_", str(iter_num), ".csv"])

    # Train the model
    train_model(model_path = MODEL_PATH, pickle_path = PICKLE_PATH, model_name = MODEL_NAME, batch_size = BATCH_SIZE, 
        epochs = EPOCH_COUNT, normalize_labels = NORMALIZE_LABELS, train = train, validation = validation, 
        use_validation = USE_VALIDATION, validation_split = VALIDATION_SPLIT, model_performance_file = model_performance_file, full = USE30, verbose = True)


# %%
