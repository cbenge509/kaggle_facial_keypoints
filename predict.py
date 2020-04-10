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
USE30 = True

# Processing behavior (constants)
AVAILABLE_MODELS = ["KERAS_LENET5", "KERAS_INCEPTION", "KERAS_KAGGLE1", "KERAS_NAIMISHNET", "KERAS_CONVNET5", "KERAS_INCEPTIONV3", "KERAS_KAGGLE2", "KERAS_RESNET50", "KERAS_RESNET", "KERAS_RESNEXT50", "KERAS_RESNEXT101"]
TEST_DATA_FILE = "cleandata_naive_test.pkl"
TEST_IDS_FILE = "raw_id_lookup.pkl"
OVERLAP_FILE = "cleandata_naive_overlap.pkl"

TEST8_DATA_FILE = "cleandata_test8.pkl"
TEST30_DATA_FILE = "cleandata_test30.pkl"

#%%

############################################################################
# ARGUMENT SPECIFICATION
############################################################################

parser = argparse.ArgumentParser(description = "Performs predictions for the Kaggle Facial Keypoints Detection challenge.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-pp', '--pickle_path', type = str, default = "C:/kaggle/kaggle_keypoints/pickle", help = "Path to location of output pickle files (post processing files).")
parser.add_argument('-mp', '--model_path', type = str, default = "C:/kaggle/kaggle_keypoints/models", help = "Path to location of output model files.")
parser.add_argument('-m', '--model_name', type = str, default = "KERAS_LENET5", help = "Name of the model to train.")
parser.add_argument('-pa', '--partial', action = 'store_true', help = 'Trains only using the 8-value dataset (vs. the full 30-value dataset)')
parser.add_argument('-nl', '--normalize_labels', action = 'store_true', help = "Enables the normalization of prediction label values prior to training.")

############################################################################
# ARGUMENT PARSING
############################################################################


def process_arguments(parsed_args, display_args = False):
    
    global VERBOSE, PICKLE_PATH, MODEL_PATH, MODEL_NAME, NORMALIZE_LABELS, USE30
    
    args = vars(parser.parse_args())

    if display_args:
        print("".join(["\PREDICT Arguments in use:\n", "-" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    USE30 = not args['partial']
    MODEL_NAME = args['model_name']
    NORMALIZE_LABELS = args['normalize_labels']

    MODEL_PATH = str(args['model_path']).lower().strip().replace('\\', '/')
    PICKLE_PATH = str(args['pickle_path']).lower().strip().replace('\\', '/')

    # validate the presence of the paths
    for p, v, l in zip([MODEL_PATH, PICKLE_PATH], ['model_path', 'pickle_path'], ['Model file path', 'Pickle file path']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    # validate the parameters entered
    if not MODEL_NAME in AVAILABLE_MODELS:
        raise RuntimeError("Parameter `model_name` value of '%s' is invalid.  Must be in list: %s" % (MODEL_NAME, str(AVAILABLE_MODELS)))


#%%

############################################################################
# LOAD DATA
############################################################################

# load the data for training
def load_data(pickle_path, test_file, id_file, overlap_file, verbose = True):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN LOAD DATA <<<\n", "-" * 50, "\n"]))

    if not pickle_path.endswith("/"): pickle_path = "".join([pickle_path, "/"])
    test_file = "".join([pickle_path, test_file])
    id_file = "".join([pickle_path, id_file])
    overlap_file = "".join([pickle_path, overlap_file])

    for f, l in zip([test_file, id_file, overlap_file], ['Test', 'Test IDs', 'Overlap']):
        if not os.path.isfile(f):
            raise RuntimeError("%s file '%s' not found - training cancelled." % (l, f))

    test = pickle.load(open(test_file, "rb"))
    if verbose: print("Test file '%s' loaded; shape: %s" % (test_file, str(test.shape)))

    ids = pickle.load(open(id_file, "rb"))
    if verbose: print("Test IDs file '%s' loaded; shape: %s" % (id_file, str(ids.shape)))

    overlap = pickle.load(open(overlap_file, "rb"))
    if verbose: print("Overlap file '%s' loaded; shape: %s" % (overlap_file, str(overlap.shape)))

    if verbose: print("".join(["\n", "-" * 50, "\n>>> END LOAD DATA <<<\n", "-" * 50, "\n"]))

    return test, ids, overlap

# %%
############################################################################
# PREDICT MODEL (GENERIC HANDLER)
############################################################################

def predict_model(model_path, pickle_path, model_name, normalize_labels, test, ids, overlap, predict_file, skip_output = False, skip_overlap = False, full = True, verbose = True):

    if verbose: print("".join(["-" * 50, "\n>>> BEGIN PREDICT ON %s <<<\n" % model_name, "-" * 50, "\n"]))

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

    # call the training module specific to the algorithm called
    if model_name == "KERAS_LENET5":
        feature_name = "ALL_FEATURES"
        pred = predict_model_lenet5(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_INCEPTIONV3":
        feature_name = "ALL_FEATURES"
        pred, _ = predict_model_inceptionv3(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_RESNET50":
        feature_name = "ALL_FEATURES"
        pred = predict_model_resnet50(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_RESNEXT50":
        feature_name = "ALL_FEATURES"
        pred = predict_model_resnext50(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_RESNEXT101":
        feature_name = "ALL_FEATURES"
        pred = predict_model_resnext101(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_RESNET":
        feature_name = "ALL_FEATURES"
        pred = predict_model_resnet(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_INCEPTION":
        feature_name = "ALL_FEATURES"
        Y_main, Y_aux1, Y_aux2, Y_main_cols, Y_aux1_cols, Y_aux2_cols =  predict_model_inception(models = models, 
            xform = xform, test = test, ids = ids, feature_name = feature_name, full = full, verbose = verbose)
        pred = [Y_main, Y_aux1, Y_aux2, Y_main_cols, Y_aux1_cols, Y_aux2_cols]

    elif model_name == "KERAS_KAGGLE1":
        feature_name = "ALL_FEATURES"
        pred = predict_model_kaggle1(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_KAGGLE2":
        feature_name = "ALL_FEATURES"
        pred = predict_model_kaggle2(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_CONVNET5":
        feature_name = "ALL_FEATURES"
        pred = predict_model_convnet5(models = models, xform = xform, test = test, ids = ids, 
            feature_name = feature_name, full = full, verbose = verbose)

    elif model_name == "KERAS_NAIMISHNET":
        if full:
            feature_name = ['left_eye_center', 'right_eye_center', 'left_eye_inner_corner', 'left_eye_outer_corner', 
                'right_eye_inner_corner', 'right_eye_outer_corner', 'left_eyebrow_inner_end', 'left_eyebrow_outer_end', 
                'right_eyebrow_inner_end', 'right_eyebrow_outer_end', 'nose_tip', 'mouth_left_corner', 'mouth_right_corner', 
                'mouth_center_top_lip', 'mouth_center_bottom_lip']
        else:
            feature_name = ['left_eye_center', 'right_eye_center', 'nose_tip', 'mouth_center_bottom_lip']
        pred = predict_model_naimishnet(models = models, xform = xform, test = test, ids = ids, feature_name = feature_name,
            normalize_labels = normalize_labels, full = full, verbose = verbose)
    else:
        raise RuntimeError("Model name '%s' not understood; cancelling training." % model_name)

    if not skip_output:
        # this branch for normal output against TEST
        output_prediction(model_path = model_path, model_name = model_name, Y = pred, test = test, ids = ids, feature_name = feature_name, 
            predict_file = predict_file, xform = xform, overlap = overlap, normalize_labels = normalize_labels, skip_overlap = skip_overlap, full = full, verbose = verbose)
    else:
        # this branch for output of STACK cross validation
        output_stack(model_path = model_path, model_name = model_name, Y = pred, test = test, ids = ids, feature_name = feature_name, 
            predict_file = predict_file, xform = xform, overlap = overlap, normalize_labels = normalize_labels, skip_overlap = skip_overlap, full = full, verbose = verbose)

    if verbose: print("".join(["-" * 50, "\n>>> END PREDICT ON %s <<<\n" % model_name, "-" * 50, "\n"]))

    return pred
# %%

############################################################################
# PREDICT MODEL NAIMISHNET
############################################################################

def predict_model_naimishnet(models, xform, test, ids, feature_name, normalize_labels, full = True, verbose = True):

    # create empty DF for capturing inferenced values (unpivoted x,y coordinates to columns)
    submission = pd.DataFrame({'image_id':int(), 'variable':'', 'value':float()},index=[1])
    submission = submission[(submission.index == -1)]

    df = {}
    for keypoint in feature_name:
        X, subset = xform.PrepareTest(test, ids, keypoint, verbose = verbose)
        subset = subset[['image_id']]
        Y = models.predict_keras_naimishnet(X = X, feature_name = keypoint, full = full, verbose = verbose)

        # un-normalize the predictions
        mod_subset = subset.copy()
        for i, lbl in zip(range(Y.shape[1]), ['_x', '_y']):
            if normalize_labels:
                Y[:,i] = xform.UnNormalize_Labels(Y[:,i])
            # ensure pixel boundaries are clipped between 0.0 and 96.0
            Y[:,i] = np.clip(Y[:,i], 0.0, 96.0)

            col = "".join([keypoint, lbl])
            
            mod_subset[col] = Y[:,i]

        submission = submission.append(pd.melt(mod_subset, id_vars = ['image_id']), ignore_index = True)
    submission.columns = ['image_id','feature_name','location']

    return submission

#%%

############################################################################
# PREDICT MODEL LENET5
############################################################################

def predict_model_lenet5(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_lenet5(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL KAGGLE1
############################################################################

def predict_model_kaggle1(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_kaggle1(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL KAGGLE2
############################################################################

def predict_model_kaggle2(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_kaggle2(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL INCEPTIONV3
############################################################################

def predict_model_inceptionv3(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y, Y_cols = models.predict_keras_inceptionv3(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y, Y_cols

#%%

############################################################################
# PREDICT MODEL RESNET
############################################################################

def predict_model_resnet(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_resnet(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL RESNET50
############################################################################

def predict_model_resnet50(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_resnet50(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL RESNEXT50
############################################################################

def predict_model_resnext50(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_resnext50(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL RESNEXT101
############################################################################

def predict_model_resnext101(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_resnext101(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL CONVNET5
############################################################################

def predict_model_convnet5(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test = test, ids = ids, feature_name = feature_name, verbose = verbose)
    Y = models.predict_keras_convnet5(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y

#%%

############################################################################
# PREDICT MODEL INCEPTION
############################################################################

def predict_model_inception(models, xform, test, ids, feature_name, full = True, verbose = True):

    X, _ = xform.PrepareTest(test, ids, feature_name, verbose = verbose)
    Y_main, Y_aux1, Y_aux2, Y_main_cols, Y_aux1_cols, Y_aux2_cols = models.predict_keras_inception(X = X, feature_name = feature_name, full = full, verbose = verbose)

    return Y_main, Y_aux1, Y_aux2, Y_main_cols, Y_aux1_cols, Y_aux2_cols

############################################################################
# OUTPUT PREDICTIONS (STACK)
############################################################################

def output_stack(model_path, model_name, Y, feature_name, test, ids, predict_file, xform, overlap, normalize_labels, skip_overlap = False, full = True, verbose = True):

    if full:
        train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
            'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
            'right_eye_inner_corner_y', 'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
    else:
        train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 
            'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
    
    # generate output for LeNet, Kaggle1, Kaggle2, ConvNet, InceptionV3, and ResNet50
    if model_name in ['KERAS_LENET5', 'KERAS_KAGGLE1', 'KERAS_KAGGLE2', 'KERAS_CONVNET5', 'KERAS_INCEPTIONV3', 'KERAS_RESNET50', 'KERAS_RESNET', 'KERAS_RESNEXT50', 'KERAS_RESNEXT101']:

        Y = pd.DataFrame(Y, columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
        Y.index.rename('image_id', inplace = True)

        # write the predictions file
        Y.to_csv(predict_file, index = True)
        print("Predictions written to '%s'." % predict_file)
    
    elif model_name == 'KERAS_INCEPTION':

        created_files, blend_vals = [], None
        for j, l, cols in zip([Y[0], Y[1], Y[2]], ['main_model', 'aux1_model', 'aux2_model'], [Y[3], Y[4], Y[5]]):
            for ncol, col in enumerate(cols):
                #ol = overlap.copy()
                #print(l, col)
                __loop_pred_file = predict_file.replace("".join([model_name, "/"]), "".join([model_name, "/", l.upper(), "__"])).replace(".csv", "".join(["_", col.replace("/", "_"), ".csv"]))
                created_files.append(__loop_pred_file)
                j_df = pd.DataFrame(j[ncol], columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
                j_df.index.rename('image_id', inplace = True)

                for c in [c for c in train_cols if not 'image' == c]:
                    if NORMALIZE_LABELS:
                        vals = xform.UnNormalize_Labels(j_df[c].values)
                        j_df[c] = vals
                    j_df[c] = np.clip(j_df[c], 0.0, 96.0)

                if blend_vals is None:
                    blend_vals = j_df.values
                else:
                    blend_vals = np.mean((blend_vals, j_df.values), axis = 0)

                # write the predictions file
                #j_df.to_csv(__loop_pred_file, index = True)
                #print("Predictions written to '%s'." % __loop_pred_file)

        # now iterate over all the created files and create a blend
        df_combined = pd.DataFrame(blend_vals, columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
        df_combined.index.rename('image_id', inplace = True)

        df_combined.to_csv(predict_file, index = True)
        print("\nBlended predictions written to '%s' (mean average of all %d Inception model predictions).\n\n" % (predict_file, len(created_files)))

    elif model_name == "KERAS_NAIMISHNET":

        df = {}
        df['image_id'] = test.image_id.values
        for c in [c for c in train_cols if not 'image' == c]:
            df[c] = Y[(Y.image_id.isin(test.image_id.values) & (Y.feature_name == c))].location.values
        
        df = pd.DataFrame(df).set_index('image_id')

        df.to_csv(predict_file, index = True)
        print("Predictions written to '%s'." % predict_file)

    return


#%%

############################################################################
# OUTPUT PREDICTIONS (TEST)
############################################################################

def output_prediction(model_path, model_name, Y, feature_name, test, ids, predict_file, xform, overlap, normalize_labels, skip_overlap = False, full = True, verbose = True):

    if full:
        train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
            'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x',
            'right_eye_inner_corner_y', 'right_eye_outer_corner_x','right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
            'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y',
            'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
            'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
    else:
        train_cols = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y', 
            'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']

    # generate output for LENET5, CONVNET5, KAGGLE1, KAGGLE2, INCEPTIONV3, RESNET50, and for our stacking METAREGRESSOR_LINEAR
    if model_name in ['KERAS_LENET5', 'KERAS_KAGGLE1', 'KERAS_KAGGLE2', 'KERAS_CONVNET5', 'METAREGRESSOR_LINEAR', 'KERAS_INCEPTIONV3', 'KERAS_RESNET50', 'KERAS_RESNET', 'KERAS_RESNEXT50', 'KERAS_RESNEXT101']:

        Y = pd.DataFrame(Y, columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
        Y = pd.melt(Y.reset_index(), id_vars=['index'])
        Y.columns = ['image_id', 'feature_name','location']
        Y = ids.drop(columns=['location']).merge(Y, on=['image_id','feature_name'], how = 'inner').drop(columns=['image_id','feature_name'])
        Y.columns = ['RowId','Location']

        if normalize_labels:
            norm_Y = xform.UnNormalize_Labels(Y.Location.values)
            Y.Location = norm_Y

        Y.Location = np.clip(Y.Location, 0.0, 96.0)

        # write the predictions file
        Y.to_csv(predict_file, index = False)
        print("Predictions written to '%s'." % predict_file)

        if not skip_overlap:
            # write the predictions w/ overlap file
            overlap = pd.melt(overlap, id_vars=['image_id'])
            overlap.columns = ['image_id', 'feature_name','location']
            overlap = overlap.merge(ids.drop(columns=['location']), on = ['image_id','feature_name'], how = 'inner')
            overlap = overlap[['row_id', 'location']]
            overlap.columns = ['RowId', 'Location']

            Y = Y.set_index('RowId')
            overlap = overlap.set_index('RowId')
            Y.update(overlap, join = 'left', overwrite = True)
            Y = Y.reset_index()
            Y.to_csv(predict_file.replace("predictions", "OVERLAPpredictions"), index = False)
            print("Overlap predictions written to '%s'." % predict_file.replace("predictions", "OVERLAPpredictions"))
    
    elif model_name == 'KERAS_INCEPTION':

        created_files = []
        for j, l, cols in zip([Y[0], Y[1], Y[2]], ['main_model', 'aux1_model', 'aux2_model'], [Y[3], Y[4], Y[5]]):
            for ncol, col in enumerate(cols):
                ol = overlap.copy()
                #print(l, col)
                __loop_pred_file = predict_file.replace("".join([model_name, "/"]), "".join([model_name, "/", l.upper(), "__"])).replace(".csv", "".join(["_", col.replace("/", "_"), "_output",".csv"]))
                created_files.append(__loop_pred_file)
                j_df = pd.DataFrame(j[ncol], columns = [c for c in train_cols if not 'image' == c], index = test.image_id.values)
                j_df = pd.melt(j_df.reset_index(), id_vars=['index'])
                j_df.columns = ['image_id', 'feature_name','location']
                j_df = ids.drop(columns=['location']).merge(j_df, on=['image_id','feature_name'], how = 'inner').drop(columns=['image_id','feature_name'])
                j_df.columns = ['RowId','Location']

                if NORMALIZE_LABELS:
                    vals = xform.UnNormalize_Labels(j_df.Location.values)
                    j_df.Location = vals

                j_df.Location = np.clip(j_df.Location, 0.0, 96.0)

                # write the predictions file
                j_df.to_csv(__loop_pred_file, index = False)
                print("Predictions written to '%s'." % __loop_pred_file)

                if not skip_overlap:
                    # write the predictions w/ overlap file
                    ol = pd.melt(ol, id_vars=['image_id'])
                    ol.columns = ['image_id', 'feature_name','location']
                    ol = ol.merge(ids.drop(columns=['location']), on = ['image_id','feature_name'], how = 'inner')
                    ol = ol[['row_id', 'location']]
                    ol.columns = ['RowId', 'Location']

                    j_df = j_df.set_index('RowId')
                    ol = ol.set_index('RowId')
                    j_df.update(ol, join = 'left', overwrite = True)
                    j_df = j_df.reset_index()
                    j_df.to_csv(__loop_pred_file.replace("predictions", "OVERLAPpredictions"), index = False)
                    print("Overlap predictions written to '%s'." % __loop_pred_file.replace("predictions", "OVERLAPpredictions"))

        # now iterate over all the created files and create a blend
        df_combined, col_num = None, 0
        for f in created_files:
            df = pd.read_csv(f)
            col_num += 1
            if (df_combined is None):
                df_combined = df[df.index == -1]
                df_combined['RowId'] = df.RowId
            df_combined[str(col_num)] = df.Location
        df_combined = df_combined.set_index('RowId')
        df_combined['Location'] = df_combined.mean(axis = 1)
        df_combined = df_combined.reset_index()
        df_combined = df_combined[['RowId','Location']]
        df_combined.to_csv(predict_file, index = False)
        print("\n\nBlended predictions written to '%s' (average of all %d model results).\n\n" % (predict_file, len(created_files)))

        if not skip_overlap:
            # write the blended predictions file with overlaps
            overlap = pd.melt(overlap, id_vars=['image_id'])
            overlap.columns = ['image_id', 'feature_name','location']
            overlap = overlap.merge(ids.drop(columns=['location']), on = ['image_id','feature_name'], how = 'inner')
            overlap = overlap[['row_id', 'location']]
            overlap.columns = ['RowId', 'Location']

            df_combined = df_combined.set_index('RowId')
            overlap = overlap.set_index('RowId')
            df_combined.update(overlap, join = 'left', overwrite = True)
            df_combined = df_combined.reset_index()
            df_combined.to_csv(predict_file.replace("predictions", "OVERLAPpredictions"), index = False)
            print("Predictions written to '%s'." % predict_file.replace("predictions", "OVERLAPpredictions"))

    elif model_name == "KERAS_NAIMISHNET":

        # now will need to do an inner join with df_ids to get the row_id
        Y = Y.merge(ids[['image_id','feature_name','row_id']], on = ['image_id','feature_name'], how = 'inner')
        Y = Y[['row_id','location']]

        # write the predictions file
        if VERBOSE: print("Writing predictions out to disk...")
        Y.columns = ['RowId','Location']
        pd.DataFrame(Y).to_csv(predict_file, index = False)
        print("Predictions written to '%s'." % predict_file)

        if not skip_overlap:
            # write the predictions w/ overlap file
            overlap = pd.melt(overlap, id_vars=['image_id'])
            overlap.columns = ['image_id', 'feature_name','location']
            overlap = overlap.merge(ids.drop(columns=['location']), on = ['image_id','feature_name'], how = 'inner')
            overlap = overlap[['row_id', 'location']]
            overlap.columns = ['RowId', 'Location']

            Y = Y.set_index('RowId')
            overlap = overlap.set_index('RowId')
            Y.update(overlap, join = 'left', overwrite = True)
            Y = Y.reset_index()
            Y.to_csv(predict_file.replace("predictions", "OVERLAPpredictions"), index = False)
            print("Overlap predictions written to '%s'." % predict_file.replace("predictions", "OVERLAPpredictions"))

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
    if USE30: TEST_DATA_FILE = TEST30_DATA_FILE
    else: TEST_DATA_FILE = TEST8_DATA_FILE

    test, ids, overlap = load_data(pickle_path = PICKLE_PATH, test_file = TEST_DATA_FILE, 
        id_file = TEST_IDS_FILE, overlap_file = OVERLAP_FILE, verbose = VERBOSE)

    # derive the model predict file location
    output_path = str(MODEL_PATH).replace("\\", "/").strip()
    if not output_path.endswith('/'): output_path = "".join((output_path, "/", MODEL_NAME, "/"))
    if USE30:
        iter_num = max(([0] + [int(f.replace("predictions_30_", "")[:-4]) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and f.startswith('predictions_30_')])) + 1
        predict_file = "".join([output_path, "predictions_30_", str(iter_num), ".csv"])
    else:
        iter_num = max(([0] + [int(f.replace("predictions_8_", "")[:-4]) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and f.startswith('predictions_8_')])) + 1
        predict_file = "".join([output_path, "predictions_8_", str(iter_num), ".csv"])

    # Generate predictions
    predict_model(model_path = MODEL_PATH, pickle_path = PICKLE_PATH, model_name = MODEL_NAME, 
        normalize_labels = NORMALIZE_LABELS, test = test, ids = ids, overlap = overlap, 
        predict_file = predict_file, full = USE30, verbose = VERBOSE)

# %%
