#%%
import numpy as np
import pandas as pd
import pickle
import json

from os import listdir
from os.path import isfile, join

KEYPOINTS_PATH = "C:/kaggle/kaggle_keypoints/data/augmented_keypoints/ann/"
FILE_PREFIX = "test_"
FILE_SUFFIX = ".png.json"
TEST_UPDATE_FILE = "C:/kaggle/kaggle_keypoints/pickle/cleandata_updates_augment.pkl"

train = pickle.load(open("C:/kaggle/kaggle_keypoints/pickle/cleandata_naive_train.pkl", "rb"))

test = pd.read_csv("C:/kaggle/kaggle_keypoints/data/test.csv", names = ['image_id', 'image'], dtype = {'image_id':'uint16', 'image':'object'}, skiprows = 1)
test.image = test.image.map(lambda x: np.array(list(map(int, x.split(" ")))))

#%%

# iterate through the json label meta file and populate a dict object to obtain the guids
label_keys = {}
with open('C:/kaggle/kaggle_keypoints/data/augmented_keypoints/meta.json') as json_file:
    data = json.load(json_file)['classes'][0]['geometry_config']['nodes']
    for p in data:
        key = data[p]['label'].replace(" ", "_").lower()
        if not key == '16' and not key == '17':
            label_keys[key] = str(p)


#%%

listdir(KEYPOINTS_PATH)
files = [f for f in listdir(KEYPOINTS_PATH)]


image_keypoints = {'image_id':[], 'left_eye_center_x':[],'left_eye_center_y':[],'left_eye_outer_x':[],
            'left_eye_outer_y':[],'left_eye_inner_x':[],'left_eye_inner_y':[],'left_eyebrow_outer_x':[],
            'left_eyebrow_outer_y':[],'left_eyebrow_inner_x':[],'left_eyebrow_inner_y':[],'right_eyebrow_inner_x':[],
            'right_eyebrow_inner_y':[],'right_eye_inner_x':[],'right_eye_inner_y':[],'right_eye_center_x':[],
            'right_eye_center_y':[],'right_eye_outer_x':[],'right_eye_outer_y':[],'right_eyebrow_outer_x':[],
            'right_eyebrow_outer_y':[],'nose_tip_x':[],'nose_tip_y':[],'mouth_center_top_lip_x':[],
            'mouth_center_top_lip_y':[],'mouth_center_bottom_lip_x':[],'mouth_center_bottom_lip_y':[],
            'mouth_left_corner_x':[],'mouth_left_corner_y':[],'mouth_right_corner_x':[],'mouth_right_corner_y':[]}

# loop over files
# for now, just try one to get the JSON right
for test_file in files:

    with open("".join([KEYPOINTS_PATH, test_file])) as json_file:
        data = json.load(json_file)['objects']
        if len(data) > 0:
            image_id = np.uint32(test_file.replace(FILE_PREFIX, "").replace(FILE_SUFFIX, "").replace(".json", ""))
            kp = data[0]['nodes']
            image_keypoints['image_id'].append(image_id)
            for kp_label in label_keys.keys():
                loc = kp[label_keys[kp_label]]
                x_lbl, y_lbl = "".join([kp_label, "_x"]), "".join([kp_label, "_y"])
                image_keypoints[x_lbl].append(np.float32(loc['loc'][0]))
                image_keypoints[y_lbl].append(np.float32(loc['loc'][1]))

dtypes={'image_id':np.int64,
    'left_eye_center_x':np.float32,'left_eye_center_y':np.float32,'left_eye_outer_x':np.float32,
    'left_eye_outer_y':np.float32,'left_eye_inner_x':np.float32,'left_eye_inner_y':np.float32,
    'left_eyebrow_outer_x':np.float32,'left_eyebrow_outer_y':np.float32,'left_eyebrow_inner_x':np.float32,
    'left_eyebrow_inner_y':np.float32,'right_eyebrow_inner_x':np.float32,'right_eyebrow_inner_y':np.float32,
    'right_eye_inner_x':np.float32,'right_eye_inner_y':np.float32,'right_eye_center_x':np.float32,
    'right_eye_center_y':np.float32,'right_eye_outer_x':np.float32,'right_eye_outer_y':np.float32,
    'right_eyebrow_outer_x':np.float32,'right_eyebrow_outer_y':np.float32,'nose_tip_x':np.float32,
    'nose_tip_y':np.float32,'mouth_center_top_lip_x':np.float32,'mouth_center_top_lip_y':np.float32,
    'mouth_center_bottom_lip_x':np.float32,'mouth_center_bottom_lip_y':np.float32,'mouth_left_corner_x':np.float32,
    'mouth_left_corner_y':np.float32,'mouth_right_corner_x':np.float32,'mouth_right_corner_y':np.float32}

image_keypoints = pd.DataFrame.from_dict(image_keypoints).astype(dtypes)

rename_cols = {'left_eye_inner_x':'left_eye_inner_corner_x',
    'left_eye_inner_y':'left_eye_inner_corner_y',
    'left_eye_outer_x':'left_eye_outer_corner_x',
    'left_eye_outer_y':'left_eye_outer_corner_y',
    'left_eyebrow_inner_x':'left_eyebrow_inner_end_x',
    'left_eyebrow_inner_y':'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_x':'left_eyebrow_outer_end_x',
    'left_eyebrow_outer_y':'left_eyebrow_outer_end_y',
    'right_eye_inner_x':'right_eye_inner_corner_x',
    'right_eye_inner_y':'right_eye_inner_corner_y',
    'right_eye_outer_x':'right_eye_outer_corner_x',
    'right_eye_outer_y':'right_eye_outer_corner_y',
    'right_eyebrow_inner_x':'right_eyebrow_inner_end_x',
    'right_eyebrow_inner_y':'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_x':'right_eyebrow_outer_end_x',
    'right_eyebrow_outer_y':'right_eyebrow_outer_end_y'}
image_keypoints.rename(columns=rename_cols, inplace=True)


#%%

image_keypoints = image_keypoints.merge(test[['image_id','image']], on = 'image_id', how = 'inner').set_index('image_id')
test = test.set_index('image_id')
image_keypoints = image_keypoints[train.columns]
pickle.dump(image_keypoints, open(TEST_UPDATE_FILE, "wb"))
print("Pickle file written: '%s' with %d images..." % (TEST_UPDATE_FILE, image_keypoints.shape[0]))

# %%
