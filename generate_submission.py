#%%
############################################################################
# IMPORTS
############################################################################

import pandas as pd
import numpy as np
import pickle
import os

#%%

############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations and model name (parameters)
MODEL_PATH = "C:/kaggle/kaggle_keypoints/models"

#%%

############################################################################
# HELPER FUNCTIONS
############################################################################

def process_predictions(model_path, model_name):
    
    print("".join(["-" * 100, "\n>>> Generating Submission for %s <<<\n" % model_name, "-" * 100, "\n"]))

    full_path = "".join([os.path.join(model_path, model_name).replace('\\','/'),"/"])
    
    # Process predictions (non-overlap)
    iter_num30 = max(([0] + [int(f.replace("predictions_30_", "")[:-4]) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.startswith('predictions_30_')]))
    iter_num8 = max(([0] + [int(f.replace("predictions_8_", "")[:-4]) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.startswith('predictions_8_')]))

    for i, msg in zip([iter_num30, iter_num8], ["FULL (30)", "PARTIAL (8)"]):
        if i == 0:
            print("".join(["No ", msg, " prediction file found - skipping submission generation."]))
            return

    latest30 = os.path.join(full_path, "".join(["predictions_30_", str(iter_num30), ".csv"]))
    latest8 = os.path.join(full_path, "".join(["predictions_8_", str(iter_num8), ".csv"]))
    final_file = os.path.join(full_path, "".join(["SUBMISSION_", model_name, "__(30_", str(iter_num30), ")_(8_", str(iter_num8), ").csv"]))

    print("Merging '%s' and '%s'..." % (latest30, latest8))
    
    if model_name == 'KERAS_NAIMISHNET':
        df30 = pd.read_csv(latest30, index_col = 'RowId')
        df8 = pd.read_csv(latest8, index_col = 'RowId')

        df30.update(df8)
        df = df30.append(df8[(~(df8.index.isin(df30.index.values)))])

        df = df.sort_index(axis = 0, ascending = True).reset_index()
        df.to_csv(final_file, index = False)
    else:
        df = pd.read_csv(latest30, index_col = 'RowId')
        df = df.append(pd.read_csv(latest8, index_col = 'RowId'))
        df = df.sort_index(axis = 0, ascending = True).reset_index()
        df.to_csv(final_file, index = False)

    print("Submission generated %s: '%s'.\n" % (str(df.shape), final_file))

    # Process predictions OVERLAP
    iter_num30 = max(([0] + [int(f.replace("OVERLAPpredictions_30_", "")[:-4]) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.startswith('OVERLAPpredictions_30_')]))
    iter_num8 = max(([0] + [int(f.replace("OVERLAPpredictions_8_", "")[:-4]) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.startswith('OVERLAPpredictions_8_')]))

    for i, msg in zip([iter_num30, iter_num8], ["FULL OVERLAP (30)", "PARTIAL OVERLAP (8)"]):
        if i == 0:
            print("".join(["No ", msg, " prediction file found - skipping submission generation."]))
            return
    
    latest30 = os.path.join(full_path, "".join(["OVERLAPpredictions_30_", str(iter_num30), ".csv"]))
    latest8 = os.path.join(full_path, "".join(["OVERLAPpredictions_8_", str(iter_num8), ".csv"]))
    final_file = os.path.join(full_path, "".join(["SUBMISSION_OVERLAP_", model_name, "__(30_", str(iter_num30), ")_(8_", str(iter_num8), ").csv"]))

    print("Merging '%s' and '%s'..." % (latest30, latest8))

    if model_name == 'KERAS_NAIMISHNET':
        df30 = pd.read_csv(latest30, index_col = 'RowId')
        df8 = pd.read_csv(latest8, index_col = 'RowId')

        df30.update(df8)
        df = df30.append(df8[(~(df8.index.isin(df30.index.values)))])

        df = df.sort_index(axis = 0, ascending = True).reset_index()
        df.to_csv(final_file, index = False)
    else:
        df = pd.read_csv(latest30, index_col = 'RowId')
        df = df.append(pd.read_csv(latest8, index_col = 'RowId'))
        df = df.sort_index(axis = 0, ascending = True).reset_index()
        df.to_csv(final_file, index = False)

    print("Overlap Submission generated %s: '%s'.\n" % (str(df.shape), final_file))

    print("".join(["-" * 100, "\n>>> Submission Generation Complete for %s <<<\n" % model_name, "-" * 100, "\n"]))

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

    #process_predictions(model_path = MODEL_PATH, model_name = 'KERAS_NAIMISHNET')
    for dirName, subdirList, fileList in os.walk(MODEL_PATH):
        for subdir in subdirList:
            process_predictions(model_path = MODEL_PATH, model_name = subdir)

# %%
