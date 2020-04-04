#%%
############################################################################
# IMPORTS
############################################################################
import numpy as np
import uuid
import argparse
import pickle
import pandas as pd
from sklearn.model_selection import KFold
import train_model, predict
import os
from time import process_time

import warnings
warnings.filterwarnings("ignore")
#%%
############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Processing behavior (parameters)
MODEL_NAMES = ["KERAS_NAIMISHNET", "KERAS_CONVNET5", "KERAS_LENET5", "KERAS_KAGGLE1", "KERAS_INCEPTION"]
K_SPLITS = 5
VERBOSE = True
USE_VALIDATION = True
BATCH_SIZE = 128
EPOCH_COUNT = 300
NORMALIZE_LABELS = False
VALIDATION_SPLIT = 0.1 # note: only used if USE_VALIDATION is False
#USE30 = True

# Default file Locations and model name (parameters)
PICKLE_PATH = "C:/kaggle/kaggle_keypoints/pickle"
STACK_PATH = "C:/kaggle/kaggle_keypoints/stacking"
TRAIN_FILE = "cleandata_train_plus_augmentation.pkl"
TEST_DATA_FILE = "cleandata_naive_test.pkl"

# Competition file names and limits (constants)
VALIDATION_DATA_FILE = "validation_set.pkl"

TRAIN30_DATA_FILE = "cleandata_train30_plus_augmentation.pkl"
TRAIN8_DATA_FILE = "cleandata_train8_plus_augmentation.pkl"

TEST30_DATA_FILE = "cleandata_test30.pkl"
TEST8_DATA_FILE = "cleandata_test8.pkl"

TEST_IDS_FILE = "raw_id_lookup.pkl"
OVERLAP_FILE = "cleandata_naive_overlap.pkl"
AVAILABLE_MODELS = ["KERAS_NAIMISHNET", "KERAS_CONVNET5", "KERAS_LENET5", "KERAS_KAGGLE1", "KERAS_INCEPTION", "KERAS_INCEPTIONV3", "KERAS_KAGGLE2"]

#%%
############################################################################
# ARGUMENT SPECIFICATION
############################################################################

parser = argparse.ArgumentParser(description = "Performs data preparation for the Kaggle Facial Keypoints Detection challenge.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
#parser.add_argument('-pa', '--partial', action = 'store_true', help = 'Uses the partial dataset (8-value) instead of the full dataset (30-value) for stacking.')
parser.add_argument('-pp', '--pickle_path', type = str, default = "C:/kaggle/kaggle_keypoints/pickle", help = "Path to location of output pickle files (post processing files).")
parser.add_argument('-sp', '--stack_path', type = str, default = "C:/kaggle/kaggle_keypoints/stacking", help = "Path to location of output stacked model files.")
parser.add_argument('-k', '--k_folds', type = int, default = 5, help = "Number of CV folds to apply to each model stacked.")
parser.add_argument('-m', '--models', nargs = '+', type = str, default = ['KERAS_CONVNET5', 'KERAS_LENET5', 'KERAS_KAGGLE1', 'KERAS_INCEPTION', 'KERAS_NAIMISHNET', 'KERAS_INCEPTIONV3'], help = 'Specifies the models to be stacked.')

parser.add_argument('-nl', '--normalize_labels', action = 'store_true', help = "Enables the normalization of prediction label values prior to training.")
parser.add_argument('-bs', '--batch_size', type = int, default = 128, help = "Specifies the batch size for neural network training.")
parser.add_argument('-e', '--epochs', type = int, default = 300, help = "Specifies the total epochs for neural network training.")

parser.add_argument('-v', '--validation', action = 'store_true', help = 'Enable the use of a validation set (disables the use of a validation_split).')
parser.add_argument('-vs', '--validation_split', type = float, default = 0.1, help = 'Specifies the validation split percentage (ignored if using a validation set).')

############################################################################
# ARGUMENT PARSING
############################################################################

def process_arguments(parsed_args, display_args = False):
    
    global  MODEL_NAMES, K_SPLITS, VERBOSE, USE_VALIDATION, BATCH_SIZE, EPOCH_COUNT, NORMALIZE_LABELS, VALIDATION_SPLIT, \
            PICKLE_PATH, STACK_PATH #, USE30
    
    args = vars(parser.parse_args())

    if display_args:
        print("".join(["\TRAIN_STACK Arguments in use:\n", "-" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    #USE30 = not args['partial']
    MODEL_NAMES = args['models']
    K_SPLITS = args['k_folds']
    NORMALIZE_LABELS = args['normalize_labels']
    BATCH_SIZE = args['batch_size']
    EPOCH_COUNT = args['epochs']
    USE_VALIDATION = args['validation']
    VALIDATION_SPLIT = args['validation_split']
    STACK_PATH = str(args['stack_path']).lower().strip().replace('\\', '/')
    PICKLE_PATH = str(args['pickle_path']).lower().strip().replace('\\', '/')

    # validate the presence of the paths
    for p, v, l in zip([STACK_PATH, PICKLE_PATH], ['stack_path', 'pickle_path'], ['Stacking file path', 'Pickle file path']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    # validate the parameters entered
    assert 0 < BATCH_SIZE < 2049, "Parameter `batch_size` must be between 1 and 2,048."
    assert 0 < EPOCH_COUNT < 1000, "Parameter `epochs` must be between 1 and 1,000."
    if not USE_VALIDATION:
        assert 0.04 < VALIDATION_SPLIT < 0.51, "Parameter `validation_split` must be between 0.05 and 0.5."
    for model_name in MODEL_NAMES:
        if not model_name in AVAILABLE_MODELS:
            raise RuntimeError("Parameter `model_name` value of '%s' is invalid.  Must be in list: %s" % (model_name, str(AVAILABLE_MODELS)))

#%%

############################################################################
# KFold Stacking
############################################################################

def kfold_stack(train, validation, kfold_splits, model_names, stack_path, pickle_path, epochs, batch_size, normalize_labels, 
    use_validation, validation_split, run_sequence, full = True, verbose = False):
    
    # capture the list of label names
    label_cols = [c for c in train.columns if not 'image' == c]
    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    actual_labels = train[(train.index == -1)].copy()
    predicted_labels = {'id':[]}

    kf = KFold(n_splits = kfold_splits, random_state = 42, shuffle = False)

    for k, (train_idx, test_idx) in enumerate(kf.split(train)):
        print("".join(["\n", "=" * 50, "\nFold Iteration %d of %d\n" % (k + 1, kfold_splits), "=" * 50, "\n"]))
        
        # get our KFold split of "train" and "test" for stacking
        #stack_train = train[(train.index.values.isin(train_idx))].copy()
        #stack_test = train[(train.index.values.isin(test_idx))].copy()
        stack_train = train.iloc[train_idx].copy()
        stack_test = train.iloc[test_idx].copy()

        # capture the actual labels for the input vector
        actual_labels = actual_labels.append(stack_test)
        if k == 0:
            predicted_labels['id'] = test_idx
        else:
            predicted_labels['id'] = np.vstack((predicted_labels['id'].reshape(-1,1), test_idx.reshape(-1,1))).ravel()

        # short-circuit logic if you need to restart at specific iteration due to hang/crash
        #if k < 3:
        #    print("Skipping k == %d..." % k)
        #    continue

        # for each model, train on the large K-fold and predict on the hold-out
        for model_name in model_names:
            print("\n", "-" * 50, "\n MODEL: %s\n" % "".join([model_name,model_suffix]), "-" * 50, "\n")
            train_model.MODEL_NAME = model_name
            predict.MODEL_NAME = model_name

            # specify the prediction file
            output_path = str(stack_path).replace("\\", "/").strip()
            if not output_path.endswith('/'): output_path = "".join((output_path, "/"))
            predict_file = "".join([output_path,"STACK_", model_name, model_suffix, "_", str(k+1), "_of_", str(kfold_splits),"_", run_sequence, ".csv"])

            # derive the model performance file location
            output_path = str(stack_path).replace("\\", "/").strip()
            if not output_path.endswith('/'): output_path = "".join((output_path, "/", model_name, "/"))
            if not os.path.exists(output_path):
                print("Creating output path: '%s'." % output_path)
                os.makedirs(output_path)
            model_performance_file = "".join([output_path, "performance_", str(k), ".csv"])

            # CDB : 3/30/2020 - why did I even make use_validation an option?  It MUST be True, and we MUST pass in the kth holdout... sigh
            models, features = train_model.train_model(model_path = stack_path, pickle_path = pickle_path, model_name = model_name, batch_size = batch_size, 
                epochs = epochs, normalize_labels = normalize_labels, train = stack_train, validation = stack_test, 
                use_validation = True, validation_split = validation_split, skip_history = True, model_performance_file = model_performance_file,
                full = full, verbose = verbose)

            # dummy up a "test" dataframe as our current utility functions expect a specific format for test and ids
            temp_test = stack_test.copy()
            temp_test = temp_test.reset_index().rename(columns = {'index':'image_id'})

            image_id = temp_test.image_id.values
            feature_name = np.array(label_cols)
            temp_ids = pd.DataFrame(np.transpose([np.tile(image_id, len(feature_name)), np.repeat(feature_name, len(image_id))]), columns = ['image_id', 'feature_name'])
            temp_ids['location'] = temp_ids.image_id
            temp_ids['row_id'] = temp_ids.image_id
            temp_ids.image_id = temp_ids.image_id.astype(np.int64)
            temp_ids.row_id = temp_ids.row_id.astype(np.int64)
            temp_ids.location = temp_ids.location.astype(np.float32)
            temp_ids = temp_ids[['row_id', 'image_id', 'feature_name', 'location']]

            # for each kfold iteration, we need to predict and store the predictions
            pred = predict.predict_model(model_path = stack_path, pickle_path = pickle_path, model_name = model_name, normalize_labels = normalize_labels, 
                test = temp_test, ids = temp_ids, overlap = None, predict_file = predict_file, skip_output = True, skip_overlap = True, full = full, verbose = verbose)

    # KFold training iterations complete; write the final labels file
    print("".join(["\n", "=" * 50, "\nFolds Complete. Writing Labels\n", "=" * 50, "\n"]))
    output_path = str(stack_path).replace("\\", "/").strip()
    if not output_path.endswith('/'): output_path = "".join((output_path, "/"))
    actual_labels = actual_labels.drop(columns=['image'])
    actual_labels.index.rename('image_id', inplace = True)
    labels_file = "".join([output_path,"STACK", model_suffix, "_labels_", run_sequence, ".csv"])
    actual_labels.to_csv(labels_file)
    print("Labels file written to '%s'." % labels_file)

    return

#%%

############################################################################
# Train full models
############################################################################

def train_models(model_names, stack_path, pickle_path, batch_size, epochs, normalize_labels, use_validation, validation_split, full = True, verbose = False):

    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    print("".join(["\n", "=" * 50, "".join(["\nFull Data Training Phase", model_suffix, "\n"]), "=" * 50, "\n"]))

    for model_name in model_names:
        print("\n", "-" * 50, "\n MODEL: %s\n" % "".join([model_name, model_suffix]), "-" * 50, "\n")
        train_model.MODEL_NAME = model_name
        predict.MODEL_NAME = model_name

        # derive the model performance file location
        output_path = str(stack_path).replace("\\", "/").strip()
        if not output_path.endswith('/'): output_path = "".join((output_path, "/", model_name, "/"))
        if not os.path.exists(output_path):
            print("Creating output path: '%s'." % output_path)
            os.makedirs(output_path)
        model_performance_file = "".join([output_path, "performance", model_suffix, "_FULL.csv"])

        # train on the full dataset
        models, features = train_model.train_model(model_path = stack_path, pickle_path = pickle_path, model_name = model_name, batch_size = batch_size, 
            epochs = epochs, normalize_labels = normalize_labels, train = train, validation = validation, use_validation = use_validation, 
            validation_split = validation_split, skip_history = True, model_performance_file = model_performance_file, full = full, verbose = verbose)

        print("Model '%s' full data training complete.\n" % model_name)

    print("".join(["\n", "=" * 50, "\nFull Data Training Phase Complete\n", "=" * 50, "\n"]))

    return
#%%

############################################################################
# Predict full models
############################################################################

def predict_models(model_names, stack_path, pickle_path, normalize_labels, run_sequence, full = True, verbose = False):

    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    print("".join(["\n", "=" * 50, "\nFull Data Prediction Phase\n", "=" * 50, "\n"]))

    for model_name in model_names:
        print("\n", "-" * 50, "\n MODEL: %s\n" % "".join([model_name, model_suffix]), "-" * 50, "\n")
        predict.MODEL_NAME = model_name

        # derive the model performance file location
        output_path = str(stack_path).replace("\\", "/").strip()
        if not output_path.endswith('/'): output_path = "".join((output_path, "/"))
        if not os.path.exists(output_path):
            raise RuntimeError("Cannot predict for model '%s' - output path '%s' does not exist." % (model_name, output_path))
        predict_file = "".join([output_path, "PREDICT_", model_name, model_suffix, "_", run_sequence, ".csv"])

        # predict on full dataset trained model
        pred = predict.predict_model(model_path = stack_path, pickle_path = pickle_path, model_name = model_name, normalize_labels = normalize_labels, 
            test = test, ids = ids, overlap = overlap, predict_file = predict_file, skip_output = True, skip_overlap = False, full = full, verbose = verbose)

        print("Model '%s' full data prediction complete.\n" % model_name)

    print("".join(["\n", "=" * 50, "\nFull Data Prediction Phase Complete\n", "=" * 50, "\n"]))

    return
#%%

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == "__main__":

    start_time = process_time()

    # Clear the screen
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    
    # Process command-line arguments and set parameters
    process_arguments(parser.parse_args(), display_args = True)

    print("".join(["-" * 100, "\n>>> STACKING INITIATED <<<\n", "-" * 100, "\n"]))

    # generate a unique run sequence to avoid file creation collision
    run_sequence = str(uuid.uuid1()).replace('-','')
    #run_sequence = "570ee126724411eaa950000d3a129e03"
    
    print("Stacking started with run sequence:", run_sequence, "\n")

    # if the stacking root path doesn't exist, create it
    if not os.path.exists(STACK_PATH):
        os.makedirs(STACK_PATH)
    train_model.MODEL_PATH = STACK_PATH
    predict.MODEL_PATH = STACK_PATH

    for f, tr, te in zip([True, False], [TRAIN30_DATA_FILE, TRAIN8_DATA_FILE], [TEST30_DATA_FILE, TEST8_DATA_FILE]):
    #for f, tr, te in zip([False, True], [TRAIN8_DATA_FILE, TRAIN30_DATA_FILE], [TEST8_DATA_FILE, TEST30_DATA_FILE]):
        # load the training and validation datasets
        train, validation = train_model.load_data(pickle_path = PICKLE_PATH, train_file = tr, 
            validation_file = VALIDATION_DATA_FILE, use_validation = USE_VALIDATION, verbose = False)

        # Load the test dataset (required for prediction calls)
        test, ids, overlap = predict.load_data(pickle_path = PICKLE_PATH, test_file = te, 
            id_file = TEST_IDS_FILE, overlap_file = OVERLAP_FILE, verbose = False)

        # perform KFold stacking of N models
        kfold_stack(train = train, validation = validation, kfold_splits = K_SPLITS, model_names = MODEL_NAMES, 
            stack_path = STACK_PATH, pickle_path = PICKLE_PATH, epochs = EPOCH_COUNT, batch_size = BATCH_SIZE, normalize_labels = NORMALIZE_LABELS, 
            use_validation = USE_VALIDATION, validation_split = VALIDATION_SPLIT, verbose = VERBOSE, run_sequence = run_sequence, full = f)
        
        # train full models for use in final inferencing
        train_models(model_names = MODEL_NAMES, stack_path = STACK_PATH, pickle_path = PICKLE_PATH, batch_size = BATCH_SIZE, epochs = EPOCH_COUNT, 
            normalize_labels = NORMALIZE_LABELS, use_validation = False, validation_split = 0.1, verbose = VERBOSE, full = f)

        # predict outputs for the fully trained model (used in metaregressor prediction)
        predict_models(model_names = MODEL_NAMES, stack_path = STACK_PATH, pickle_path = PICKLE_PATH, normalize_labels = NORMALIZE_LABELS, 
            run_sequence = run_sequence, verbose = VERBOSE, full = f)
 
    # report the elapsed time
    end_time = process_time()
    #print("Elapsed time in minutes:", round((60./(end_time - start_time)),2))

    # signal the end of stacking
    print("".join(["-" * 100, "\n>>> STACKING COMPLETE <<<\n", "-" * 100, "\n"]))