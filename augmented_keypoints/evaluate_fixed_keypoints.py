#%%
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt


TRAIN_UPDATE_FILE = "C:/kaggle/kaggle_keypoints/pickle/cleandata_updates_augment.pkl"
train = pickle.load(open(TRAIN_UPDATE_FILE, "rb")).reset_index()

print("Size of 'augmentation' set: %d" % train.shape[0])

# %%

fig = plt.figure(figsize=(20,20))
cols = [c for c in train.columns if not c.startswith('image')]
rng = np.clip(train.shape[0], 0, 60)

for i in range(rng):
    img = train.iloc[i].image.reshape(96,96)
    points = train.iloc[i][cols].values
    ax = fig.add_subplot(6,10,i+1)
    ax.imshow(img, cmap='gray')
    ax.scatter(points[0::2], points[1::2], color = 'red', s = 20)
    plt.axis('off')


plt.tight_layout()
plt.show()


# %%
