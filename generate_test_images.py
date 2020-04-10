import numpy as np
from PIL import Image
import pickle

test = pickle.load(open("C:/kaggle/kaggle_keypoints/pickle/cleandata_naive_test.pkl", "rb"))

for n in range(len(test)):
    img = np.array(np.array(test.iloc[n].image * 255.).reshape(96, 96), dtype = np.uint8)
    img_id = test.iloc[n].image_id
    file_name = "".join(["C:/kaggle/test_images/test_",str(img_id),".png"])
    im = Image.fromarray(img)
    im.save(file_name)

print("test file creation done.")