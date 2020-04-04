#%%
############################################################################
# IMPORTS
############################################################################
import numpy as np
import os
import re
import argparse
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, LassoLars, MultiTaskElasticNet, MultiTaskLasso, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import predict, train_model

#%%
#%%
############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Processing behavior (parameters)
VERBOSE = True
SCALE_DATA = False
METAREGRESSOR_PREDICT_ALL = False
RUN_SEQUENCE = ""
USE_INTERACTIONS = False

# Default file Locations and model name (parameters)
PICKLE_PATH = "C:/kaggle/kaggle_keypoints/pickle"
STACK_PATH = "C:/kaggle/kaggle_keypoints/stacking"

# Competition file names and limits (constants)
#TRAIN_FILE = "cleandata_train_plus_augmentation.pkl"
#TEST_DATA_FILE = "cleandata_naive_test.pkl"

#TRAIN30_DATA_FILE = "cleandata_train30_plus_augmentation.pkl"
#TRAIN8_DATA_FILE = "cleandata_train8_plus_augmentation.pkl"

TEST30_DATA_FILE = "cleandata_test30.pkl"
TEST8_DATA_FILE = "cleandata_test8.pkl"

VALIDATION_DATA_FILE = "validation_set.pkl"
TEST_IDS_FILE = "raw_id_lookup.pkl"
OVERLAP_FILE = "cleandata_naive_overlap.pkl"
LINEAR_MODEL_NAME = "METAREGRESSOR_LINEAR"
TRAIN_COLS = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
    'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
    'right_eye_inner_corner_y', 'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
    'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
    'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
    'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

#%%
############################################################################
# ARGUMENT SPECIFICATION
############################################################################

parser = argparse.ArgumentParser(description = "Performs metaregressor prediction of stacked outputs from L1 models for the Kaggle Facial Keypoints Detection challenge.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-a', '--all', action = 'store_true', help = 'Metaregresor trains on all and predicts all dependent variables in one pass (vs. one-at-a-time)')
parser.add_argument('-pp', '--pickle_path', type = str, default = "C:/kaggle/kaggle_keypoints/pickle", help = "Path to location of output pickle files (post processing files).")
parser.add_argument('-sp', '--stack_path', type = str, default = "C:/kaggle/kaggle_keypoints/stacking", help = "Path to location of output stacked model files.")
parser.add_argument('-rs', '--run_sequence', type = str, default = "", help = "Specifies a specific stacking run sequence to train (default is to use the latest one in the stacking directory)")
parser.add_argument('-sd', '--scale_data', action = 'store_true', help = "Enables standard scaling of prediction label values prior to training.")
parser.add_argument('-i', '--interactions', action = 'store_true', help = "Creates multiplication interactions for all columns on a per-model basis (not available on OAAT strategy)")

#%%
############################################################################
# ARGUMENT PARSING
############################################################################

def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, SCALE_DATA, RUN_SEQUENCE, PICKLE_PATH, STACK_PATH, METAREGRESSOR_PREDICT_ALL, USE_INTERACTIONS

    args = vars(parser.parse_args())

    if display_args:
        print("".join(["\PREDICT_STACK Arguments in use:\n", "-" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    SCALE_DATA = args['scale_data']
    RUN_SEQUENCE = args['run_sequence']
    METAREGRESSOR_PREDICT_ALL = args['all']
    USE_INTERACTIONS = args['interactions']

    STACK_PATH = str(args['stack_path']).lower().strip().replace('\\', '/')
    PICKLE_PATH = str(args['pickle_path']).lower().strip().replace('\\', '/')

    # validate the presence of the paths
    for p, v, l in zip([STACK_PATH, PICKLE_PATH], ['stack_path', 'pickle_path'], ['Stacking file path', 'Pickle file path']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    if (not METAREGRESSOR_PREDICT_ALL) and USE_INTERACTIONS:
        raise RuntimeError("Can not use one-at-a-time prediction strategy with creation of feature interactions.")

    return

#%%
############################################################################
# CREATE TRAINING DATASET
############################################################################

def get_train(stack_path, run_sequence, scale_data, create_interactions = False, full = True, verbose = False):

    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    print("".join(["\n", "=" * 50, "".join(["\nBuilding Train Data for L2", model_suffix, "\n"]), "=" * 50, "\n"]))

    # get the latest run_sequence if one is not provided by caller
    if run_sequence == "":
        if full:
            files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and f.endswith('.csv') and f.startswith('STACK_30_labels_')]
        else:
            files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and f.endswith('.csv') and f.startswith('STACK_8_labels_')]
        if len(files) == 0: 
            raise RuntimeError("Stack labels are missing; nothing to predict.")

        # getctime depends on Windows enviornment
        dates = [datetime.fromtimestamp(os.path.getctime(os.path.join(stack_path,f))).strftime('%Y-%m-%d %H:%M:%S') for f in files]
        labels = pd.DataFrame({'file_name':files, 'date':dates})

        label_file = labels.sort_values(by='date', ascending = False).iloc[0].file_name
        run_sequence = label_file.replace('STACK_30_labels_', '').replace('STACK_8_labels_', '').replace('.csv', '')
    else:
        files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and f.endswith('.csv') and f.startswith('STACK_labels_') and run_sequence in f]
        if len(files) == 0: 
            raise RuntimeError("Stack labels are missing or run sequence '%s' doesn't exist; nothing to predict." % run_sequence)
        if full:
            label_file = os.path.join(stack_path, "".join(["STACK_30_labels_", run_sequence, ".csv"]))
        else:
            label_file = os.path.join(stack_path, "".join(["STACK_8_labels_", run_sequence, ".csv"]))

    # if the STACK output already exists for this run sequence, delete them
    files = [f for f in os.listdir(STACK_PATH) if os.path.isfile(os.path.join(STACK_PATH,f)) and f.endswith("".join([run_sequence, ".csv"])) and f.startswith('STACK__')]
    if full:
        files = [f for f in files if '_30_' in f]
    else:
        files = [f for f in files if '_8_' in f]
    for old_stack in files:
        os.remove(os.path.join(stack_path, old_stack))

    files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and not f.startswith('STACK_30_labels_') and not f.startswith('STACK_8_labels_') and run_sequence in f and f.startswith('STACK_')]
    if full:
        files = [f for f in files if '_30_' in f]
    else:
        files = [f for f in files if '_8_' in f]

    df = pd.DataFrame({'file':files})

    models = []
    for fname in df.sort_values(by = 'file', ascending = True).values:
        trim_fname = fname[0].replace("".join(["_",run_sequence, ".csv"]), '').replace('STACK_', '').replace('_30_', '_').replace('_8_', '_')
        model_name = re.sub(r"_\d{1,}_of_\d{1,}", "", trim_fname, flags = re.IGNORECASE)
        if model_name not in models: models.append(model_name)

    print("Model results to stack: %d" % len(models))
    X = pd.read_csv(os.path.join(stack_path, label_file))[['image_id']].set_index('image_id')
    labels = pd.read_csv(os.path.join(stack_path, label_file)).set_index('image_id')
    floats = [c for c in labels if labels[c].dtype == 'float64']
    labels[floats] = labels[floats].astype(np.float32)

    for model in models:
        print("Loading model files for '%s'..." % model)

        files = np.sort([f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and f.startswith("".join(["STACK_", model, "_"])) and run_sequence in f])
        if full:
            files = [f for f in files if '_30_' in f]
        else:
            files = [f for f in files if '_8_' in f]
        df = None
        for f in files:
            print("\tProcessing: %s..." % f)
            if df is None:
                df = pd.read_csv(os.path.join(stack_path, f), index_col = 'image_id')
            else:
                df = df.append(pd.read_csv(os.path.join(stack_path, f), index_col = 'image_id'))
        floats = [c for c in df if df[c].dtype == 'float64']
        df[floats] = df[floats].astype(np.float32)
        df.columns = ["".join([model, "_", c]) for c in df.columns]

        #CDB: add interactions
        if create_interactions:
            cols = df.columns
            for n_a, cola in enumerate(cols):
                for n_b, colb in enumerate(cols):
                    if n_b > n_a:
                        col_name = "".join([cola,"_interaction_", colb])
                        df[col_name] = df[cola] * df[colb]

        X = X.merge(df, how = 'left', left_index = True, right_index = True)
    
    cols = X.columns
    
    if scale_data:
        ss = StandardScaler()
        X[cols] = ss.fit_transform(X[cols].values)
    else:
        ss = None

    print("".join(["\n", "=" * 50, "".join(["\nTrain Data for L2", model_suffix, " Built\n"]), "=" * 50, "\n"]))

    return X, labels, ss, run_sequence, models

#%%

############################################################################
# Train the Metaregressor
############################################################################

def train_metaregressor(stack_path, train, labels, run_sequence, scale_data, models, predict_mode_all, full = True, verbose = False):

    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    print("".join(["\n", "=" * 50, "".join(["\nTraining Metaregressor", model_suffix, " (Level 2)\n"]), "=" * 50, "\n"]))

    # Model definition for metaregressor
    if predict_mode_all:
        model = MultiTaskElasticNet(random_state = 42, max_iter = 1000, l1_ratio = 1.0, alpha = 0.1)
    else:
        model = ElasticNet(random_state = 42, max_iter = 1000, l1_ratio = 1.0, alpha = 0.1)
    
    print('Training linear metaregressors for %d models and %d total independent variables.\n' % (len(models), train.shape[1]))
    
    reg_models, rmse = [], []
    if predict_mode_all:
        print("// MODE: All-in-One Pass //\n")
        model.fit(train.values, labels.values)
        rmse = [np.sqrt(mean_squared_error(y_true = labels.values, y_pred = model.predict(train.values)))]
        reg_models.append(model)
    else:
        print("// MODE: One-at-a-Time //\n")
        # iterate and build a model over all dependent variables (30)
        for f in range(len(TRAIN_COLS)):
            # get the list of values to predict, column-wise
            predict_me = labels.values[:,f]
            # build the list of independent variables 
            for i in range((0+f), ((30 * len(models)) + f), 30):
                if i == 0+f:
                    train_me = train.values[:,i].reshape(-1, 1)
                else:
                    train_me = np.hstack((train_me, train.values[:,i].reshape(-1, 1)))
            # fit and store in our reg_models list
            model.fit(train_me, predict_me)
            reg_models.append(model)
            score = np.sqrt(mean_squared_error(y_true = predict_me, y_pred = model.predict(train_me)))
            rmse.append(score)
            print("Metaregressor #%d of %d trained for feature '%s'; RMSE was: %.5f" % 
                ((f + 1), len(TRAIN_COLS), TRAIN_COLS[f], score))
    
    print("\nAll metaregressors trained; average RMSE: %.5f" % np.mean(rmse))

    print("".join(["\n", "=" * 50, "".join(["\nMetaregressor", model_suffix, " Training Complete\n"]), "=" * 50, "\n"]))

    return reg_models
#%%

############################################################################
# Make Final (L2) Predictions
############################################################################

def predict_level2(stack_path, reg, ss, test, ids, overlap, scale_data, run_sequence, models, labels, predict_mode_all, create_interactions = False, full = True, verbose = False):

    if full: model_suffix = "_30"
    else: model_suffix = "_8"

    # filter the ids list to only those under consideration (since we've split this call up into full and partial TEST list (30/8))
    ids = ids[(ids.image_id.isin(test.image_id.values))]

    print("".join(["\n", "=" * 50, "".join(["\nMetaregressor", model_suffix, " (L2) Predictions\n"]), "=" * 50, "\n"]))

    if (not run_sequence) or str(run_sequence).strip() == "":
        raise RuntimeError("run_sequence must be a valid sequence value to continue.")
    
    files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path,f)) and f.endswith("".join([run_sequence, ".csv"])) and f.startswith('PREDICT_')]
    if len(files) == 0: 
        raise RuntimeError("Stack prediction files (PREDICT_...) are missing; nothing for metaregressor to predict.")

    X = test.copy()[['image_id']].set_index('image_id')
    print("Stacking full-model predictions for the following models:", ", ".join(models).strip())
    for model in models:
        if full:
            model_fname = "".join(["PREDICT_", model, "_30_", run_sequence, ".csv"])
        else:
            model_fname = "".join(["PREDICT_", model, "_8_", run_sequence, ".csv"])
        if not model_fname in files:
            raise RuntimeError("Prediction file '%s' is missing; cancelling metaregressor predictions." % model_fname)
        df = pd.read_csv(os.path.join(stack_path, model_fname), index_col = 'image_id')
        floats = [c for c in df if df[c].dtype == 'float64']
        df[floats] = df[floats].astype(np.float32)
        df.columns = ["".join([model, "_", c]) for c in df.columns]

        #CDB: add interactions
        if create_interactions:
            cols = df.columns
            for n_a, cola in enumerate(cols):
                for n_b, colb in enumerate(cols):
                    if n_b > n_a:
                        col_name = "".join([cola,"_interaction_", colb])
                        df[col_name] = df[cola] * df[colb]

        X = X.merge(df, how = 'left', left_index = True, right_index = True)
    
    # scale input vector, if required
    if scale_data:
        X[TRAIN_COLS] = ss.transform(X[TRAIN_COLS].values)
    
    # generate metaregressor predictions from stacked input (1783, 30) with image_id as index column
    #pred = reg.predict(X.values)

    if predict_mode_all:
        pred = reg[0].predict(X.values)
    else:
        pred = None
        for f in range(len(reg)):
            # retrieve the model for the given feature 'f'
            model = reg[f]
            # for this given model, find the data in X to predict against
            for i in range((0+f), ((30 * len(models)) + f), 30):
                if i == 0+f:
                    predict_me = X.values[:,i].reshape(-1, 1)
                else:
                    predict_me = np.hstack((predict_me, X.values[:,i].reshape(-1, 1)))
            # predict 
            if pred is None:
                pred = model.predict(predict_me).reshape(-1, 1)
            else:
                pred = np.hstack((pred, model.predict(predict_me).reshape(-1, 1)))


    predict_file = os.path.join(stack_path, "".join(["STACK__", model_suffix,"_","_".join(models), "__predictions__", run_sequence, ".csv"])).replace('\\', '/')

    # generate the metaregressor prediction
    predict.output_prediction(model_path = stack_path, model_name = LINEAR_MODEL_NAME, Y = pred, feature_name = "ALL_FEATURES", full = full,
        test = test, ids = ids, predict_file = predict_file, xform = None, overlap = overlap, normalize_labels = False, skip_overlap = False, verbose = verbose)

    print("".join(["\n", "=" * 50, "".join(["\nMetaregressor", model_suffix, " (L2) Predictions Complete\n"]), "=" * 50, "\n"]))

    return

#%%

############################################################################
# FINAL SUBMISSION GENERATOR
############################################################################

def generate_submission(stack_path, run_sequence, verbose = False):

    # create the final stacking files
    files = [f for f in os.listdir(stack_path) if os.path.isfile(os.path.join(stack_path, f)) and f.endswith("".join([run_sequence, ".csv"])) and f.startswith('STACK__')]
    base_files = [f for f in files if '_predictions_' in f]
    overlap_files = [f for f in files if '_OVERLAPpredictions_' in f]

    # name the final output files as "SUBMISSION_STACK__..."
    base_output_file = base_files[0].replace('_30_', '').replace('_8_', '').replace('STACK__', 'SUBMISSION_STACK__')
    base_overlap_file = overlap_files[0].replace('_30_', '').replace('_8_', '').replace('STACK__', 'SUBMISSION_STACK__')

    # build the base files submission
    for fn, files in zip([base_output_file, base_overlap_file], [base_files, overlap_files]):
        df = None
        for f in files:
            if df is None:
                df = pd.read_csv(os.path.join(stack_path, f), index_col = 'RowId')
            else:
                df = df.append(pd.read_csv(os.path.join(stack_path, f), index_col = 'RowId'))
        df = df.sort_index(axis = 0, ascending = True).reset_index()
        df.to_csv(os.path.join(stack_path, fn), index = False)
        if verbose: print("'%s' written to disk - shape: %s" % (fn, str(df.shape)))

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

    print("".join(["-" * 100, "\n>>> METAREGRESSOR PREDICTION INITIATED <<<\n", "-" * 100, "\n"]))

    # if the stacking root path doesn't exist, throw error
    if not os.path.exists(STACK_PATH):
        raise RuntimeError("Stacking file path '%s' does not exist." % STACK_PATH)
    predict.MODEL_PATH = STACK_PATH

    for f, te in zip([True, False], [TEST30_DATA_FILE, TEST8_DATA_FILE]):
        # get the training data, standard scaler (if applicable), and the run sequence [if derived]
        train, labels, ss, run_sequence, models = get_train(stack_path = STACK_PATH, run_sequence = RUN_SEQUENCE, 
            scale_data = SCALE_DATA, create_interactions = USE_INTERACTIONS, full = f, verbose = VERBOSE)

        # train the metaregressor
        print("Train shape: %s" % str(train.shape))
        print("labels shape: %s" % str(labels.shape))
        print("models:", models)

        reg = train_metaregressor(stack_path = STACK_PATH, train = train, labels = labels, run_sequence = run_sequence, 
            scale_data = SCALE_DATA, models = models, predict_mode_all = METAREGRESSOR_PREDICT_ALL, full = f, verbose = VERBOSE)

        # Load the test dataset (required for prediction calls)
        test, ids, overlap = predict.load_data(pickle_path = PICKLE_PATH, test_file = te, 
            id_file = TEST_IDS_FILE, overlap_file = OVERLAP_FILE, verbose = VERBOSE)

        # make predictions with metaregressor and save predcitions in submission format
        predict_level2(stack_path = STACK_PATH, reg = reg, ss = ss, test = test, ids = ids, overlap = overlap,
            scale_data = SCALE_DATA, run_sequence = run_sequence, models = models, labels = labels, 
            predict_mode_all = METAREGRESSOR_PREDICT_ALL, create_interactions = USE_INTERACTIONS, full = f, verbose = VERBOSE)

    # generate final submission files
    generate_submission(stack_path = STACK_PATH, run_sequence = run_sequence, verbose = VERBOSE)

    # signal the end of stacking
    print("".join(["-" * 100, "\n>>> METAREGRESSOR PREDICTION COMPLETE <<<\n", "-" * 100, "\n"]))

#%%