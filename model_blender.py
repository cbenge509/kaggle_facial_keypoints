#%%
# Blend our latest predictions using unweighted mean averaging
import pandas as pd
import numpy as np
import os
import glob
from os import listdir
from os.path import isfile, join
from scipy import stats


MODEL_PATH = "C:/kaggle/kaggle_keypoints/models/" 

_, MODELS, _ = next(os.walk(MODEL_PATH))
#MODELS = ["KERAS_KAGGLE1", "KERAS_LENET5", "KERAS_INCEPTION", "KERAS_NAIMISHNET", "KERAS_INCEPTIONV3", "KERAS_CONVNET5"]
#MODELS = ["KERAS_KAGGLE1", "KERAS_NAIMISHNET"]

# should we use the base or overlap predictions?
USE_OVERLAP = False

# Clear the screen
if os.name == 'nt':
    _ = os.system('cls')
else:
    _ = os.system('clear')

print("".join(["-" * 50, "\n>>> BEGIN BLENDING <<<\n", "-" * 50, "\n"]))

# find the latest prediction and files
files, model_dict = [], {}
output_path = str(MODEL_PATH).replace("\\", "/").strip()
if not output_path.endswith('/'): output_path = "".join((output_path, "/"))

for model in MODELS:
    
    model_output_path, file_prefix = "".join([output_path, model, "/"]), ""

    if not USE_OVERLAP:    
        list_of_files = glob.glob("".join([model_output_path, "SUBMISSION_", model, "*.csv"]))
    else:
        list_of_files = glob.glob("".join([model_output_path, "SUBMISSION_OVERLAP_", model, "*.csv"]))
    
    if len(list_of_files) > 0:
        model_dict[model] = []
        latest_submission_file = max(list_of_files, key = os.path.getctime)
        files.append(os.path.join(model_output_path, latest_submission_file))

# capture the "Location" prediction for each TEST row from each model above...
arr, df, col = None, None, -1
for f in files:
    col += 1
    temp = pd.read_csv(f)
    temp_arr = temp.Location.values.reshape(-1, 1)
    model_dict[list(model_dict.keys())[col]] = temp_arr
    if df is None: df = temp[['RowId']]
    if arr is None: 
        arr = temp_arr
    else:
        arr = np.hstack((arr, temp_arr))
#%%

keys = list(model_dict.keys())
for i in range(0, len(keys)):
    for n in range(i, len(keys)):
        if not (i == n):
            base_model = keys[i]
            compare_model = keys[n]
            corr = stats.pearsonr(np.array(model_dict[base_model]).ravel(), np.array(model_dict[compare_model]).ravel())[0]
            print("Pearson r between [%s] and [%s] == %.10f" % (base_model, compare_model, corr))

# mean average the predicted Location values
df['Location'] = np.mean(arr, axis = 1).astype(np.float32)

# write file out to a blend prediction
df.to_csv("".join([output_path, "BLENDprediction.csv"]), index = False)
print("\nBLEND file written to %s.\n" % "".join([output_path, "BLENDprediction.csv"]))

print("".join(["-" * 50, "\n>>> END BLENDING <<<\n", "-" * 50, "\n"]))

# %%