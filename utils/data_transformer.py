##############################################
### Imports
##############################################

import pandas as pd
import numpy as np
import zlib
import pickle
import os.path
import cv2
from math import sin, cos, pi
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import time
from scipy import ndimage
from skimage.exposure import rescale_intensity, equalize_adapthist

##############################################
### Constants
##############################################

KERNEL_SHARPEN = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape((3, 3, 1))
KERNEL_BLUR_SMALL = np.array(np.ones((7, 7), dtype = "float") * (1. / (7 * 7))).reshape((7, 7, 1))
KERNEL_LAPLACIAN = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]).reshape((3, 3, 1))


##############################################
### Class Definition
##############################################

class Xform(object):

    def __init__(self, pickle_path, verbose = False):
        
        # validate that the constructor parameters were provided by caller
        if (not pickle_path):
            raise RuntimeError('path to pickle files must be provided on initialization.')
        
        # ensure all are string snd leading/trailing whitespace removed
        pickle_path = str(pickle_path).replace('\\', '/').strip()
        if (not pickle_path.endswith('/')): pickle_path = ''.join((pickle_path, '/'))

        # validate the existence of the data path
        if (not os.path.isdir(pickle_path)):
            raise RuntimeError("Pickle path specified'%s' is invalid." % pickle_path)

        self.__pickle_path = pickle_path


    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    # Private Methods
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    ##############################################
    ### CLEAN DUPLICATES IN TRAIN
    ##############################################
    def __remove_dupe_images(self, df, override_set = None, verbose = False):
        """Remove duplicate images from TRAIN; mean averages their x,y coordinates"""
        # expects that the only columns are the x,y coordinates and the image column

        # move index to column value for joins
        df = df.reset_index()
        # create a no-collision hash for the training data (verified no collisions manually on 2/20/2020)
        df['hash_image'] = df.image.map(lambda x: zlib.adler32(x))

        # identify duplicates
        df_dupes_hash = pd.DataFrame(df.groupby(by='hash_image').index.count().sort_values()).reset_index()
        df_dupes_hash.columns = ['hash_image', 'frequency']
        df_dupes_hash = df_dupes_hash[(df_dupes_hash.frequency > 1)]
        # join back to train to get the index value in the original dataframe
        df_dupes_hash = pd.merge(df_dupes_hash, df[['index', 'hash_image']],  \
            how = 'left', on=['hash_image']).sort_values(by=['frequency', 'hash_image'], ascending = False)
        # remove the hash from the original train dataframe now that we have it in the dupes dataframe
        df.drop(columns=['hash_image'], inplace=True)

        if verbose:
            print("Unique images that are duplicated in TRAIN: [%d].\nTotal rows affected: [%d]." % \
                (len(np.unique(df_dupes_hash.hash_image)), len(df_dupes_hash)))

        # filter to the labels and hashed image only, stored in fixed_images array
        cols = [c for c in df.columns if 'index' not in c]
        # create an empty dataframe with the proper column names and data types
        fixed_images = df[(df.index == -1)][cols].copy()

        # CDB: 3/6/2020 - added logic to allow override set to define the keypoints vs. mean averaging
        if override_set is None:
            override_present = False
        else:
            override_present = True
            override_set = override_set.reset_index()
            override_set['hash_image'] = override_set.image.map(lambda x: zlib.adler32(x))

        override_count = 0
        # create array for the label value column names only
        cols = [c for c in df.columns if c.endswith('_x') | c.endswith('_y')]
        # iterate over each unique image hash in the duplicates array
        for hash_id in df_dupes_hash.hash_image.unique():
            # get the index values for all of the images duplicates with the hash_id we are currently addressing
            df_idx = df_dupes_hash[(df_dupes_hash.hash_image == hash_id)]['index'].values
            # get the pixel array for the duplicated image
            img = df[(df['index'].isin(df_idx))].image.values[0]

            # CDB 3/6/2020 - added logic to allow for override set to define the keypoints vs. mean averaging
            if (override_present) and (hash_id in override_set.hash_image.values):
                # use the first matching value in the override_set
                fixed = pd.DataFrame(override_set[(override_set.hash_image == hash_id)].iloc[0][cols], columns = cols)
                override_count += 1
            else:
                # average the x,y corredinate labels together for all of the duplicated hash image we are addressing
                fixed = pd.DataFrame(pd.DataFrame(df[(df['index'].isin(df_idx))], columns = cols).mean(axis = 0)).T
            
            # add the image pixel array to the end of our newly averaged x,y coordinates
            fixed['image'] = [img]
            # store the new de-duped, averaged coordinates image to our array of "fixed images"
            fixed_images = fixed_images.append(fixed, ignore_index = True)

        if verbose: print("Train shape before duplicate replacement: %s" % str(df.shape))
        # drop all duplicated images from the main train dataframe and remove the index column
        df = df[~(df['index'].isin(df_dupes_hash['index'].values))]
        df.drop(columns=['index'], inplace = True)
        #if verbose: print("Train shape after truncate: %s" % str(df.shape))

        # append the fixed images to the train array and reset the index sequence
        df = df.append(fixed_images, ignore_index = True).reset_index()
        df.drop(columns=['index'], inplace = True)
        if verbose: print("Train shape after duplicate replacement: %s" % str(df.shape))
        if verbose: print("Duplication detection complete; number of images with labels overridden (instead of averaged): %d" % override_count)

        return df
    
    ##############################################
    ### SCALE IMAGE PIXEL VALUES
    ##############################################
    def __scale_image_pixels(self, df, verbose = False):
        """Scales pixel values for the 'image' column in the user provided 'df' dataframe by dividing pixel values by 255.0"""
        
        if verbose: print("Pixel scaling initiated on %d images..." % df.shape[0])
        if 'image' in df.columns:
            image_scaled = df.image.values
            image_scaled = image_scaled / 255.0
            df.image = image_scaled
            #df.image = df.image.map(lambda x: np.array([v / 255.0 for v in x]))
            if verbose: print("Pixel values scaled for %d observations." % df.shape[0])
        else:
            print("Pixel scaling skipped as user-provided dataframe does not contain column 'image'")
        
        return df

    ##############################################
    ### RETURN THE OVERLAPPING OBSERVATIONS 
    ### IN BOTH TRAIN AND TEST
    ##############################################
    def __return_overlap_with_labels(self, train, test, verbose = False):
        """Returns the intersection of TRAIN and TEST dataframes with indexes and labels"""

        # there is a chance of capturing "extra" overlap due to augmentation
        # here, we'll limit the index values of our copy of train between 0 and 1708 (1709 total rows)
        train_copy = train.copy()
        train_copy = train_copy[(train_copy.index.values < 1709)]

        # calculate a low-collision CRC for the images in both test and train
        train_copy['hash_image'] = train_copy.image.map(lambda x: zlib.adler32(x))
        test['hash_image'] = test.image.map(lambda x: zlib.adler32(x))

        cols = [c for c in train_copy.columns if c.endswith('_x') | c.endswith('_y')]
        cols.append('hash_image')

        overlap = test.merge(train_copy[cols], on = 'hash_image', how = 'inner')
        
        # drop hash image
        overlap.drop(columns = ['image','hash_image'], inplace = True)
        train_copy.drop(columns = ['hash_image'], inplace = True)
        test.drop(columns = ['hash_image'], inplace = True)

        return overlap

    ##############################################
    ### SYMMETRIC LINEAR FOLD OF 1D AXIS
    ##############################################
    def __symmetric_1Dfold(self, x, xmin = 0, xmax = 95):
        """returns the mirrored / symmetric reposition of some value 'x' along a line bounded by xmin and xmax"""
        
        # find mid-point of array
        ### midpoint = ((xmax - xmin) / 2.)
        ### shift = -2 * (x - midpoint)
        
        # move the x value symmetric to the mid-point
        ### return np.clip(((x + shift) - 1), np.float(xmin), np.float(xmax))

        # wow, I really overthought that approach above...
        return np.clip(np.float(xmax - xmin) - x, np.float(xmin), np.float(xmax))

    ##############################################
    ### PERFORM HORIZONTAL FLIP OF IMAGES W/ LABELS 
    ##############################################
    def __horizontal_flip(self, train, verbose = False):
        """Returns the horizontally flipped images of TRAIN with labels adjusted"""

        augmented = train.copy()
        
        ## CDB: 2/23/2020 limit flipping to only those with all keypoints present
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]

        # horizontally flip the images
        augmented.image = augmented.image.map(lambda x: np.flip(x.reshape(96,96), axis=1).ravel())

        cols = [c for c in augmented.columns if c.endswith('_x')]
        
        # shift all 'x' values by linear mirroring
        for c in cols:
            mod = augmented[c].values
            mod = self.__symmetric_1Dfold(mod)
            augmented[c] = mod

        # CDB 3/5/2020 : relabel the right and left's!
        cols = augmented.columns
        rename_cols = {
            'left_eye_inner_corner_x':'right_eye_inner_corner_x',
            'left_eye_center_x':'right_eye_center_x',
            'left_eye_outer_corner_x':'right_eye_outer_corner_x',
            'left_eyebrow_inner_end_x':'right_eyebrow_inner_end_x',
            'left_eyebrow_outer_end_x':'right_eyebrow_outer_end_x',
            'mouth_left_corner_x':'mouth_right_corner_x',
            'right_eye_inner_corner_x':'left_eye_inner_corner_x',
            'right_eye_center_x':'left_eye_center_x',
            'right_eye_outer_corner_x':'left_eye_outer_corner_x',
            'right_eyebrow_inner_end_x':'left_eyebrow_inner_end_x',
            'right_eyebrow_outer_end_x':'left_eyebrow_outer_end_x',
            'mouth_right_corner_x':'mouth_left_corner_x'}
        augmented.rename(columns=rename_cols, inplace=True)
        
        # change the column order back to original
        augmented = augmented[cols]

        if verbose: print("Horizontal image-flip dataframe created; size %s" % str(augmented.shape))
        return augmented

    ##############################################
    ### PERFORM +/- IMAGE ROTATIONS
    ##############################################
    def __rotate_images(self, train, rotation_angles = [12], verbose = False):
        """Rotate images by specified degrees"""

        augmented = train.copy()
        
        ## CDB: 2/23/2020 limit rotation to only those with all keypoints present
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]
        
        cols = [c for c in augmented.columns if c.endswith('_x') | c.endswith('_y')]
        
        keypoints = augmented[cols].values
        images = []
        rotated_images = []
        rotated_keypoints = []

        for i in range(len(augmented.image.values)):
            images.append(augmented.image.values[i].reshape(96,96,1))

        # Rotation augmentation for a list of angle values
        for angle in rotation_angles:
            for angle in [angle,-angle]:
                M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
                # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
                angle_rad = -angle*pi/180. 
                # For train_images
                for image in images:
                    rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                    rotated_images.append(rotated_image)
                # For train_keypoints
                for keypoint in keypoints:
                    rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                    for idx in range(0,len(rotated_keypoint),2):
                        # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                        rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                        rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                    rotated_keypoint += 48.   # Add the earlier subtracted value
                    rotated_keypoints.append(rotated_keypoint)

        rotated_images = np.array(rotated_images)
        rotated_images = rotated_images.reshape(rotated_images.shape[0], -1)
        len_rot = rotated_images.shape[0]

        make_list = []
        for i in range(len_rot):
            make_list.append((np.array(rotated_images[i])))

        df = pd.DataFrame({'id':np.arange(len_rot), 'image':make_list})
        rotated_images = df.image.values
        rotated_keypoints = np.array(rotated_keypoints)

        if verbose: print("Rotation images created; observations added to train: %d" % rotated_images.shape[0])
        return rotated_images, rotated_keypoints

    ##############################################
    ### PERFORM BRIGHT/DIM ON IMAGES
    ##############################################
    def __brighten_and_dim_images(self, train, bright_level = 1.2, dim_level = 0.6, verbose = False):
        """Brighten and dim images by specified level"""

        brighten = train.copy()
        
        ## CDB: 2/24/2020 limit brighten and dim to only those images with all 15 keypoints
        brighten = brighten[(brighten.isnull().sum(axis = 1) == 0)]
        dim = brighten.copy()

        if bright_level == 1.0:
            if verbose: print("Skipping image brightening step as 1.0 performs NOOP.")
        else:
            if verbose: print("Brightening %d images by %.2f..." % (brighten.shape[0], bright_level))
            brighten.image = brighten.image.map(lambda x: np.clip(x * bright_level, 0.0, 1.0))
        if dim_level == 1.0:
            if verbose: print("Skipping image dimming step as 1.0 performs NOOP.")
        else:
            if verbose: print("Dimming %d images by %.2f..." % (dim.shape[0], dim_level))
            dim.image = dim.image.map(lambda x: np.clip(x * dim_level, 0.0, 1.0))

        if (dim_level == 1.0) and (bright_level == 1.0):
            # if NOOP was performed, return an empty dataframe
            augmented = brighten[(brighten.index == -1)].copy()
        else:
            augmented = brighten.append(dim, ignore_index = True).reset_index().drop(columns=['index'])

        if verbose: print("Bright and dim images; observations added to train: %d" % augmented.shape[0])
        return augmented

    ##############################################
    ### ADD NOISE TO IMAGES
    ##############################################
    def __random_image_noise(self, train, verbose = False):
        """Add random gaussian normal noise to image pixels (-0.03 to 0.03)"""

        noisy = train.copy()
        
        ## CDB: 2/24/2020 limit noise to only those images with all 15 keypoints
        noisy = noisy[(noisy.isnull().sum(axis = 1) == 0)]

        cols = [c for c in noisy.columns if c.endswith('_x') | c.endswith('_y')]
        keypoints = noisy[cols].values
        images = []
        noisy_images = []

        for i in range(len(noisy.image.values)):
            images.append(noisy.image.values[i].reshape(96,96,1))

        if verbose: print("Adding random noise to  %d images..." % noisy.shape[0])
        for image in images:
            noisy_image = cv2.add(image, 0.008 * np.random.randn(96,96,1))
            noisy_images.append(noisy_image.reshape(96,96,1))            

        #noisy.image = noisy.image.map(lambda x: np.clip(cv2.add(x.reshape(96,96, 1), 0.008 * np.random.randn(96, 96, 1).ravel()), 0.0, 1.0))
        noisy_images = np.array(noisy_images)
        noisy_images = noisy_images.reshape(noisy_images.shape[0], -1)
        len_rot = noisy_images.shape[0]

        make_list = []
        for i in range(len_rot):
            make_list.append((np.array(noisy_images[i])))

        df = pd.DataFrame({'id':np.arange(len_rot), 'image':make_list})
        noisy_images = df.image.values
        keypoints = np.array(keypoints)

        if verbose: print("Random noise added for augmentation; observations added to train: %d" % noisy.shape[0])
        return noisy_images, keypoints

    ##############################################
    ### IMAGE PIXEL SHIFTING
    ##############################################
    def __vertical_and_horizontal_pixel_shift(self, train, pixel_shifts = [12], verbose = False):
        """Shift image vertically and horizontally by given shift values"""

        augmented = train.copy()
        
        ## CDB: 2/24/2020 limit pixel shifting to only those with all keypoints present
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]
        
        cols = [c for c in augmented.columns if c.endswith('_x') | c.endswith('_y')]
        
        shifted_images = []
        shifted_keypoints = []
        images = []

        keypoints = augmented[cols].values
        for i in range(len(augmented.image.values)):
            images.append(augmented.image.values[i].reshape(96,96,1))

        for shift in pixel_shifts:
            for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
                M = np.float32([[1,0,shift_x],[0,1,shift_y]])
                for image, keypoint in zip(images, keypoints):
                    shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                    shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
                    if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
                        shifted_images.append(shifted_image.reshape(96,96,1))
                        shifted_keypoints.append(shifted_keypoint)
        shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)


        shifted_images = np.array(shifted_images)
        shifted_images = shifted_images.reshape(shifted_images.shape[0], -1)
        len_rot = shifted_images.shape[0]

        make_list = []
        for i in range(len_rot):
            make_list.append((np.array(shifted_images[i])))

        df = pd.DataFrame({'id':np.arange(len_rot), 'image':make_list})
        shifted_images = df.image.values
        shifted_keypoints = np.array(shifted_keypoints)

        if verbose: print("Pixel shifted images created; observations added to train: %d" % shifted_images.shape[0])
        return shifted_images, shifted_keypoints

    ##############################################
    ### NORMALIZES ARRAY BETWEEN [t_min, t_max]
    ##############################################
    def __normalize_1Darray(self, arr, range_min = 0., range_max = 96., t_min = -1., t_max = 1., verbose = False):
        """scales the 'arr' array between the t_min and t_max parameter values"""

        # my method
        ## new_arr = []
        ## for i in range(len(arr)):
        ##    new_arr.append(((arr[i] - range_min) / (range_max - range_min)) * (t_max - t_min) + t_min)

        # CDB 2/23: NaimishNet method
        new_arr = []
        for i in range(len(arr)):
            new_arr.append((arr[i] - 48.)/48.)

        return np.array(new_arr)

    ##############################################
    ### UN-NORMALIZE ARRAY
    ##############################################
    def __unnormalize_1Darray(self, arr, t_min = -1., midpoint = 48., verbose = False):
        """scales the 'arr' array between the t_min and t_max parameter values"""

        # my method
        ## new_arr = []
        ## for i in range(len(arr)):
        ##    new_arr.append(((arr[i] - t_min) * midpoint))

        # CDB 2/23: NaimishNet method
        new_arr = []
        for i in range(len(arr)):
            new_arr.append( ((arr[i] * 48.) + 48.))
        
        return np.array(new_arr)

    ##############################################
    ### ELASTIC STRETCH OF SINGLE IMAGE + KEYPOINT
    ##############################################
    def __elastic_transform_image(self, image, keypoint_x, keypoint_y, alpha = 991, sigma = 8, random_state = 42, verbose = False):
        """elastic stretch images and individual x,y keypoint of shape (96,96)"""
        random_state = np.random.RandomState(random_state)

        keypoint_y, keypoint_x = np.clip(keypoint_y, 0.0, 95.0), np.clip(keypoint_x, 0.0, 95.0)

        shape = (96,96,2)
        img_skew = np.zeros(shape)
        img_skew[:, :, 0] = image  # layer 1, add grayscale 96x96 image
        # CDB 3/8 : added np.clip() as some keypoints were coming in with a value of 96 (which doesn't make a lot of sense in a (96,96) array that is 0 bound...)
        img_skew[int(keypoint_y), int(keypoint_x), 1] = 255.  # Layer 2, add keypoint
        img_skew = np.array(img_skew)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distorted_image = map_coordinates(img_skew, indices, order=1, mode='reflect')
        image = distorted_image.reshape(shape)[:,:,0].reshape(96,96)
        keypoints = distorted_image.reshape(shape)[:,:,1]

        
        new_keypoint_y, new_keypoint_x = np.where(keypoints > 0.0)
        if len(new_keypoint_y) == 0: new_keypoint_y = keypoint_y
        if len(new_keypoint_x) == 0: new_keypoint_x = keypoint_x
        new_keypoint_x = np.clip(np.mean(new_keypoint_x), 0.0, 95.0)
        new_keypoint_y = np.clip(np.mean(new_keypoint_y), 0.0, 95.0)
        
        if verbose:
            print("old kp x: %.3f, old kp y: %.3f -- new kp x: %.3f, new kp y: %.3f" % 
                (keypoint_x, keypoint_y, new_keypoint_x, new_keypoint_y))

        return image, new_keypoint_x, new_keypoint_y

    ##############################################
    ### ELASTIC STRETCH OF TRAIN IMAGES
    ### (one alpha + sigma combination at a time)
    ##############################################
    def __elastic_transform(self, train, alpha = 991, sigma = 8, verbose = False):
        """Returns the horizontally flipped images of TRAIN with labels adjusted"""

        augmented = train.copy()
            
        ## limit augmentation to only those with all keypoints provided
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]
        cols = [c for c in augmented.columns if not c.startswith('image')]

        if verbose: print("Stretch augmenting %d images with alpha = %d, sigma = %d..." % 
            (augmented.shape[0], alpha, sigma))
        start_time = time.time()

        stretch_images, stretch_keypoints = [], []

        for i in range(augmented.shape[0]):

            image = augmented.iloc[i].image.reshape(96,96)
            keypoints = augmented.iloc[i][cols].values

            new_img, new_keypoints = np.zeros((96,96)), []
            for i in range(0, len(keypoints), 2):
                new_img, new_kp_x, new_kp_y = self.__elastic_transform_image(image, int(keypoints[i]), 
                    int(keypoints[i+1]), alpha = alpha, sigma = sigma)
                new_keypoints.extend([new_kp_x, new_kp_y])

            stretch_images.append(new_img.ravel())
            stretch_keypoints.append(new_keypoints)
        
        df = pd.DataFrame({'id':np.arange(len(stretch_images)), 'image':stretch_images})
        stretch_images = df.image.values
        stretch_keypoints = np.array(stretch_keypoints)
        end_time = time.time()
        if verbose: print("Stretch augmentation of %d images completed in %.3f seconds (alpha = %d, sigma = %d)." % 
            (stretch_images.shape[0], (end_time - start_time), alpha, sigma))
        
        return stretch_images, stretch_keypoints

    ##############################################
    ### CONTRAST STRETCHING
    ##############################################

    def __constrast_stretching(self, train, alpha = 0.0, beta = 1.2, verbose = False):
        """perform simple contrast stretching on images in train (assumes scaling has already taken place)"""

        augmented = train.copy()
        ## limit augmentation to only those with all keypoints provided
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]

        if verbose: print("Contrast stretching %d images between alpha [%.2f] and [%.2f]..." % 
            (augmented.shape[0], alpha, beta))
        # contrast stretch each image
        images = []
        for i in range(augmented.shape[0]):
            img = augmented.iloc[i].image.reshape(96,96,1)
            img = cv2.normalize(src = img, dst = None, alpha = alpha, beta = beta, norm_type = cv2.NORM_MINMAX, dtype = -1)
            images.append(img.reshape(96,96,1))

        # transform it back into the image column
        images = np.array(images)
        images = images.reshape(images.shape[0], -1)
        
        make_list = []
        for i in range(images.shape[0]):
            make_list.append((np.array(images[i])))

        df = pd.DataFrame({'image':make_list})
        augmented.image = df.image.values
        
        # return the augmented set
        if verbose: print("Contrast stretching complete.  Returning %d observations to the augmented set." % augmented.shape[0])
        return augmented

    ##############################################
    ### APPLY KERNEL TO IMAGES
    ## Used for sharpening, blurring, smoothing, etc.
    ##############################################

    def __apply_kernel(self, train, num_times = 1, kernel = KERNEL_SHARPEN, method = "sharpen", verbose = False):
        """performs image manipulation through kernel convolution"""

        augmented = train.copy()
        ## limit augmentation to only those with all keypoints provided
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]

        if verbose: print("Applying '%s' kernel method to %d images (%d time(s))..." % 
            (method, augmented.shape[0], num_times))

        images = []
        for i in range(augmented.shape[0]):
            img = augmented.iloc[i].image.reshape(96,96,1)
            img = np.clip(np.uint8(img * 255.0), 0, 255)
            for n in range(0, num_times):
                img = ndimage.convolve(img, kernel, mode = 'nearest')
                #img = np.clip(ndimage.convolve(img, kernel, mode = 'nearest'), 0, 255).astype(np.uint8)
            img = rescale_intensity(image = img, in_range = (0, 255))
            img = img / 255.0
            images.append(img)

        images = np.array(images)
        images = images.reshape(images.shape[0], -1)

        make_list = []
        for i in range(images.shape[0]):
            make_list.append((np.array(images[i])))

        df = pd.DataFrame({'image':make_list})
        augmented.image = df.image.values

        # return the augmented set
        if verbose: print("Image '%s' complete.  Returning %d observations to the augmented set." % (method, augmented.shape[0]))
        return augmented

    ##############################################
    ### APPLY KERNEL TO IMAGES
    ## Used for sharpening, blurring, smoothing, etc.
    ##############################################

    def __apply_kernel(self, train, num_times = 1, kernel = KERNEL_SHARPEN, method = "sharpen", verbose = False):
        """performs image manipulation through kernel convolution"""

        augmented = train.copy()
        ## limit augmentation to only those with all keypoints provided
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]

        if verbose: print("Applying '%s' kernel method to %d images (%d time(s))..." % 
            (method, augmented.shape[0], num_times))

        images = []
        for i in range(augmented.shape[0]):
            img = augmented.iloc[i].image.reshape(96,96,1)
            img = np.clip(np.uint8(img * 255.0), 0, 255)
            for n in range(0, num_times):
                img = ndimage.convolve(img, kernel, mode = 'nearest')
                #img = np.clip(ndimage.convolve(img, kernel, mode = 'nearest'), 0, 255).astype(np.uint8)
            img = rescale_intensity(image = img, out_range = (0, 255))
            img = img / 255.0
            images.append(img)

        images = np.array(images)
        images = images.reshape(images.shape[0], -1)

        make_list = []
        for i in range(images.shape[0]):
            make_list.append((np.array(images[i])))

        df = pd.DataFrame({'image':make_list})
        augmented.image = df.image.values

        # return the augmented set
        if verbose: print("Image '%s' complete.  Returning %d observations to the augmented set." % (method, augmented.shape[0]))
        return augmented

    ##############################################
    ### APPLY ADAPTIVE HISTOGRAM EQUALIZATION
    ##############################################

    def __CLAHE(self, train, clip_limit = 0.03, verbose = False):
        """performs image Contrast Limitied Adaptive Histogram Equalization (CLAHE)"""

        augmented = train.copy()
        ## limit augmentation to only those with all keypoints provided
        augmented = augmented[(augmented.isnull().sum(axis = 1) == 0)]

        if verbose: print("Applying Contrast Limited Adaptive Historgram Equalization (CLAHE) to %d images using %.3f clipping limit..." % 
            (augmented.shape[0], clip_limit))

        images = []
        for i in range(augmented.shape[0]):
            img = augmented.iloc[i].image.reshape(96,96)
            #img = np.clip(np.uint8(img * 255.0), 0, 255)
            img = equalize_adapthist(img, clip_limit = clip_limit, nbins = 256)  # use default kernel_size which is 1/8 height by 1/8 width
            #img = rescale_intensity(image = img, in_range = (0.0, 1.0))
            #img = img / 255.0
            images.append(img.reshape(96,96,1))

        images = np.array(images)
        images = images.reshape(images.shape[0], -1)

        make_list = []
        for i in range(images.shape[0]):
            make_list.append((np.array(images[i])))

        df = pd.DataFrame({'image':make_list})
        augmented.image = df.image.values

        # return the augmented set
        if verbose: print("CLAHE augmentation complete.  Returning %d observations to the augmented set." % (augmented.shape[0]))
        return augmented
        

    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    # Public Methods
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    #------------------------------------------------------------------

    ##############################################
    ### Data Cleaning
    ##############################################
    def CleanData(self, train, test, verbose = False, recalculate_pickle = False, fix_labels = False, drop_missing = False, ids = None):
        """Clean Data pipeline processing - pass in TRAIN, TEST  Returns: TRAIN, TEST, OVERLAP"""
        
        # pickle file names for this pipeline
        __PICKLE_FILE_TRAIN = "".join((self.__pickle_path, "cleandata_naive_train.pkl"))
        __PICKLE_FILE_TEST = "".join((self.__pickle_path, "cleandata_naive_test.pkl"))
        __PICKLE_FILE_OVERLAP = "".join((self.__pickle_path, "cleandata_naive_overlap.pkl"))
        __PICKLE_FILE_FIXED = "".join((self.__pickle_path, "cleandata_updates_train.pkl"))
        __PICKLE_FILE_AUGMENTED = "".join((self.__pickle_path, "cleandata_updates_augment.pkl"))

        __PICKLE_FILE_TRAIN30 = "".join((self.__pickle_path, "cleandata_train30.pkl"))
        __PICKLE_FILE_TRAIN8 = "".join((self.__pickle_path, "cleandata_train8.pkl"))
        __PICKLE_FILE_TEST30 = "".join((self.__pickle_path, "cleandata_test30.pkl"))
        __PICKLE_FILE_TEST8 = "".join((self.__pickle_path, "cleandata_test8.pkl"))

        __PICKLE_FILE_VALIDATION_SET = "".join((self.__pickle_path, "validation_set.pkl"))

        # Bad images to drop (images that don't contain faces or are of dubious quality)
        __BAD_IMAGES = [6492, 6493]

        # Load TRAIN data
        if (not os.path.isfile(__PICKLE_FILE_TRAIN)) or recalculate_pickle:
            if verbose: print("Pickle file for TRAIN not found or skipped by caller. Processing...")

            # ensure only base columns present on call
            cols = [c for c in train.columns if c.endswith('_x') | c.endswith('_y')]
            cols.append('image')
            train = train[cols]
            
            # drop bad images
            if verbose: print("Dropping %d bad images from TRAIN." % len(__BAD_IMAGES))
            train = train.drop(__BAD_IMAGES)

            # Load TRAIN LABEL FIXES / AUGMENTATION data
            override_set = train[(train.index == -1)].copy()

            if fix_labels:
                if os.path.isfile(__PICKLE_FILE_FIXED):
                    if verbose: print("Pickle file for TRAIN FIXES found; Fixes being loading...")
                    image_keypoints = pickle.load(open(__PICKLE_FILE_FIXED, "rb"))
                    override_set = override_set.append(image_keypoints, ignore_index = True).reset_index().drop(columns=['index'])

                    # drop existing TRAIN records with fixed record indices
                    train = train[(~train.index.isin(image_keypoints.index.values))]

                    # append the fixes and re-sort the index of TRAIN
                    train = train.append(image_keypoints, ignore_index = False).sort_index(ascending = True)

                    if verbose: print("[%d] Fixes applied to the TRAIN dataset." % image_keypoints.shape[0])
                else:
                    if verbose: print("TRAIN FIXES skipped by user or pickle file not found.")

                if os.path.isfile(__PICKLE_FILE_AUGMENTED):
                    if verbose: print("Pickle file for TRAIN AUGMENTATION found; Additional images being loading...")
                    image_augments = pickle.load(open(__PICKLE_FILE_AUGMENTED, "rb"))
                    override_set = override_set.append(image_augments, ignore_index = True).reset_index().drop(columns=['index'])

                    # append the fixes and re-sort the index of TRAIN
                    cols = train.columns
                    train = train.append(image_augments, ignore_index = True).reset_index().sort_index(ascending = True)
                    train = train[cols]
                    
                    # write out the augmented set as a validation set!
                    if verbose: print("Writing out VALIDATION set to '%s'" % __PICKLE_FILE_VALIDATION_SET)
                    image_augments = self.__scale_image_pixels(image_augments, verbose = False)
                    pickle.dump(image_augments, open(__PICKLE_FILE_VALIDATION_SET, "wb"))

                    if verbose: print("[%d] augmentations applied to the TRAIN dataset." % image_augments.shape[0])
                else:
                    if verbose: print("TRAIN AUGMENTATION skipped by user or pickle file not found.")

            # CDB 3/6/2020 : added notion of an 'override' set to avoid averaging augmented labels with other, potentially bad labels
            # remove the duplicate images from train, average over the x,y coordinates

            #if verbose: print("Dupe removal override_set shape: %s" % str(override_set.shape))
            train = self.__remove_dupe_images(train, override_set = override_set, verbose = verbose)

            # scale pixel values in train from [0-255] to [0.0 to 1.0]
            train = self.__scale_image_pixels(train, verbose = verbose)

            # HANDLE MISSING LABELS
            # use the average values instead for the "all features" models
            # CDB : 3/27/2020 - no longer need this as we're going to split the problem into two datasets
            #if not drop_missing:
            #    for c in [col for col in train.columns if not col == 'image']:
            #        train[c] = train[c].fillna(np.nanmean(train[c].values))
            #else:
            #    shape_before = train.shape[0]
            #    train = train.dropna()
            #    if verbose: print("Dropped train with missing labels - Rows before (%d), Rows after (%d)" % (shape_before, train.shape[0]))

            #write train pickle file
            if verbose: print("Writing %d observations to TRAIN pickle file to '%s'..." % (train.shape[0], __PICKLE_FILE_TRAIN))
            pickle.dump(train, open(__PICKLE_FILE_TRAIN, "wb"))
        else:
            if verbose: print("Loading TRAIN from pickle file '%s'" % __PICKLE_FILE_TRAIN)
            train = pickle.load(open(__PICKLE_FILE_TRAIN, "rb"))
            if verbose: print("\tTRAIN shape: %s" % str(train.shape))

        # Load TEST data
        if (not os.path.isfile(__PICKLE_FILE_TEST)) or recalculate_pickle:
            if verbose: print("Pickle file for TEST not found or skipped by caller.")
            
            # normalize pixel values for test
            test = self.__scale_image_pixels(test, verbose = verbose)
            
            #write test pickle file
            if verbose: print("Writing TEST pickle file to '%s'..." % __PICKLE_FILE_TEST)
            pickle.dump(test, open(__PICKLE_FILE_TEST, "wb"))
        else:
            if verbose: print("Loading TEST from pickle file '%s'" % __PICKLE_FILE_TEST)
            test = pickle.load(open(__PICKLE_FILE_TEST, "rb"))
            if verbose: print("\tTEST shape: %s" % str(test.shape))

        # Load OVERLAP data : get the overlapping rows between train and test
        if (not os.path.isfile(__PICKLE_FILE_OVERLAP)) or recalculate_pickle:
            if verbose: print("Pickle file for OVERLAP not found or skipped by caller.")
            overlap = self.__return_overlap_with_labels(train, test)

            #write overlap pickle file
            if verbose: print("Writing OVERLAP pickle file to '%s'..." % __PICKLE_FILE_OVERLAP)
            pickle.dump(overlap, open(__PICKLE_FILE_OVERLAP, "wb"))
        else:
            if verbose: print("Loading OVERLAP from pickle file '%s'" % __PICKLE_FILE_TEST)
            overlap = pickle.load(open(__PICKLE_FILE_OVERLAP, "rb"))
            if verbose: print("\tOVERLAP shape: %s" % str(overlap.shape))

        # CDB: 3/27/2020 - split TRAIN and TEST into two problem spaces (all 30 vs. 8 only)
        if (not os.path.isfile(__PICKLE_FILE_TRAIN30)) or (not os.path.isfile(__PICKLE_FILE_TRAIN8)) or \
                (not os.path.isfile(__PICKLE_FILE_TEST30)) or (not os.path.isfile(__PICKLE_FILE_TEST8)) or \
                recalculate_pickle:
            if verbose: print("Pickle file not found for at least one of the 30/8 TRAIN or TEST files or skipped by caller.")

            # split train into dataframe containing images with all 30 dependent variables present (15 keypoints)
            train30 = train[(train.isnull().sum(axis=1) == 0)]
            # split train into dataframe containing images will ONLY 8 dependent variables present (4 keypoints)
            train8 = train[(train.isnull().sum(axis=1) == 22)]
            # limit the columns on the training labels to only those that are present
            #       ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'nose_tip_x', 'nose_tip_y',
            #        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image']
            train8_cols = train8.isnull().sum(axis=0).reset_index()[(train8.isnull().sum(axis=0).reset_index()[0] == 0)]['index'].values
            train8 = train8[train8_cols]

            # split test into prediction for those with > 8 points and those with <= 8 points
            df = ids.groupby(by='image_id').feature_name.count().reset_index()
            pred8_image_id = df[(df.feature_name <= 8)].image_id.unique()

            test8 = test[(test.image_id.isin(pred8_image_id))]
            test30 = test[~(test.image_id.isin(pred8_image_id))]

            #write the split train and test files
            if verbose: print("Writing TRAIN30 pickle file to '%s' - shape: %s" % (__PICKLE_FILE_TRAIN30, str(train30.shape)))
            pickle.dump(train30, open(__PICKLE_FILE_TRAIN30, "wb"))
            if verbose: print("Writing TRAIN8 pickle file to '%s' - shape: %s" % (__PICKLE_FILE_TRAIN8, str(train8.shape)))
            pickle.dump(train8, open(__PICKLE_FILE_TRAIN8, "wb"))
            if verbose: print("Writing TEST30 pickle file to '%s' - shape: %s" % (__PICKLE_FILE_TEST30, str(test30.shape)))
            pickle.dump(test30, open(__PICKLE_FILE_TEST30, "wb"))
            if verbose: print("Writing TEST8 pickle file to '%s' - shape: %s" % (__PICKLE_FILE_TEST8, str(test8.shape)))
            pickle.dump(test8, open(__PICKLE_FILE_TEST8, "wb"))
        else:
            if verbose: print("Loading TRAIN30 from pickle file '%s'" % __PICKLE_FILE_TRAIN30)
            train30 = pickle.load(open(__PICKLE_FILE_TRAIN30, "rb"))
            if verbose: print("\tTRAIN30 shape: %s" % str(train30.shape))

            if verbose: print("Loading TRAIN8 from pickle file '%s'" % __PICKLE_FILE_TRAIN8)
            train8 = pickle.load(open(__PICKLE_FILE_TRAIN8, "rb"))
            if verbose: print("\tTRAIN8 shape: %s" % str(train8.shape))

            if verbose: print("Loading TEST30 from pickle file '%s'" % __PICKLE_FILE_TEST30)
            test30 = pickle.load(open(__PICKLE_FILE_TEST30, "rb"))
            if verbose: print("\tTEST30 shape: %s" % str(test30.shape))

            if verbose: print("Loading TEST8 from pickle file '%s'" % __PICKLE_FILE_TEST8)
            test8 = pickle.load(open(__PICKLE_FILE_TEST8, "rb"))
            if verbose: print("\tTEST8 shape: %s" % str(test8.shape))

        return train, test, overlap, train30, train8, test30, test8

    ##############################################
    ### Data Augmentation Services
    ##############################################

    def AugmentData(self, train, horizontal_flip = False, rotation = True, rotation_angles = [9], bright_and_dim = True,
        bright_level = [1.2], dim_level = [0.6], shifting = False, pixel_shifts = [12], add_noise = True, verbose = False, 
        recalculate_pickle = False, stretch = True, stretch_alpha = [991], stretch_sigma = [8], contrast_stretch = True, 
        contrast_alpha = [0.0], contrast_beta = [1.2], sharpen = True, sharpen_rounds = [1], blur = False, blur_rounds = [1],
        smooth = False, smooth_rounds = [1], adaptive_histogram = False, adaptive_histogram_clip_limit = [0.03], full = True):
        """Provides data augmentation dataframe back to caller"""

        # pickle file names for this pipeline
        if full:
            __PICKLE_FILE_TRAIN_AUGMENT = "".join((self.__pickle_path, "augmentation_train30.pkl"))
        else:
            __PICKLE_FILE_TRAIN_AUGMENT = "".join((self.__pickle_path, "augmentation_train8.pkl"))

        # Create train augmentation dataframe
        if (not os.path.isfile(__PICKLE_FILE_TRAIN_AUGMENT)) or recalculate_pickle:
            if verbose: print("Pickle file for TRAIN AUGMENTATION not found or skipped by caller.")
            
            # create a blank dataframe
            augmented = train[(train.index == -1)]

            if horizontal_flip:
                # horizontally flip the images and adjust the labels accordingly
                if verbose: print("Augmenting TRAIN through horizontal image and label flipping.")
                augmented = augmented.append(self.__horizontal_flip(train, verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])
                
                # CDB : 3/5 for flips, we're going to treat those as "pristene" and load them into TRAIN rather than augment so augmentations are applied to them
                # note: hz flip just doesn't seem to be very helpful for this challenge; removed.
                #train = train.append(self.__horizontal_flip(train, verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])

            if sharpen:
                # apply sharpen kernel
                for rounds in sharpen_rounds:
                    if verbose: print("Augmenting TRAIN through %d rounds of image sharpening." % rounds)
                    #augmented = augmented.append(self.__sharpen_images(train, num_times = rounds, verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])
                    augmented = augmented.append(self.__apply_kernel(train, num_times = rounds, kernel = KERNEL_SHARPEN, method = "sharpen", verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])

            if blur:
                # apply blur kernel
                for rounds in blur_rounds:
                    if verbose: print("Augmenting TRAIN through %d rounds of image blurring." % rounds)
                    augmented = augmented.append(self.__apply_kernel(train, num_times = rounds, kernel = KERNEL_BLUR_SMALL, method = "blur", verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])

            if smooth:
                # apply blur kernel
                for rounds in smooth_rounds:
                    if verbose: print("Augmenting TRAIN through %d rounds of image smoothing (Laplacian)." % rounds)
                    augmented = augmented.append(self.__apply_kernel(train, num_times = rounds, kernel = KERNEL_LAPLACIAN, method = "laplacian smooth", verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])

            if adaptive_histogram:
                # apply adaptive histogram (CLAHE) augmentation 
                for clip_limit in adaptive_histogram_clip_limit:
                    if verbose: print("Augmenting TRAIN through Contrast Limited Adaptive Histogram Equalization (CLAHE) using %.3f clip limit." % clip_limit)
                    augmented = augmented.append(self.__CLAHE(train, clip_limit = clip_limit, verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])

            if rotation:
                # perform rotation of images
                if verbose: print("Augmenting TRAIN through image rotations.")
                rotated_images, rotated_keypoints = self.__rotate_images(train, rotation_angles = rotation_angles, verbose = verbose)
                rotated_aug = train[(train.index == -1)].copy()
                cols = [c for c in rotated_aug.columns if not c == 'image']
                rotated_aug[cols] = rotated_keypoints
                rotated_aug.image = rotated_images
                augmented = augmented.append(rotated_aug, ignore_index = True).reset_index().drop(columns=['index'])

            if bright_and_dim:
                # brighten and dim an augmentation set
                for b, d in zip(bright_level, dim_level):
                    if verbose: print("Brightening and dimming TRAIN images for augmentation.")
                    augmented = augmented.append(self.__brighten_and_dim_images(train, bright_level = b, dim_level = d, verbose = verbose), ignore_index = True).reset_index().drop(columns=['index'])
            
            if shifting:
                # shift pixel images vertically and horizontally
                if verbose: print("Shifting image pixels of TRAIN images for augmentation.")
                shifted_images, shifted_keypoints = self.__vertical_and_horizontal_pixel_shift(train, pixel_shifts = pixel_shifts, verbose = verbose)
                shifted_aug = train[(train.index == -1)].copy()
                cols = [c for c in shifted_aug.columns if not c == 'image']
                shifted_aug[cols] = shifted_keypoints
                shifted_aug.image = shifted_images
                augmented = augmented.append(shifted_aug, ignore_index = True).reset_index().drop(columns=['index'])

            if add_noise:
                # adding random normal noise to images
                if verbose: print("Adding Gaussian standard normal noise to image pixels (-0.03 to 0.03).")
                noisy_images, noisy_keypoints = self.__random_image_noise(train, verbose = verbose)
                noisy_aug = train[(train.index == -1)].copy()
                cols = [c for c in noisy_aug.columns if not c == 'image']
                noisy_aug[cols] = noisy_keypoints
                noisy_aug.image = noisy_images
                augmented = augmented.append(noisy_aug, ignore_index = True).reset_index().drop(columns=['index'])
            
            if stretch:
                # elastic stretch of image based on an alpha / sigma
                for a, s in zip(stretch_alpha, stretch_sigma):
                    if verbose: print("Elastic stretching TRAIN images for augmentation (alpha = %d, sigma = %d)." % (a, s))
                    stretch_images, stretch_keypoints = self.__elastic_transform(train, alpha = a, sigma = s, verbose = verbose)
                    stretch_aug = train[(train.index == -1)].copy()
                    cols = [c for c in stretch_aug.columns if not c == 'image']
                    stretch_aug[cols] = stretch_keypoints
                    stretch_aug.image = stretch_images
                    augmented = augmented.append(stretch_aug, ignore_index = True).reset_index().drop(columns=['index'])
            
            if contrast_stretch:
                # perform contrast stretching based on alpha / beta
                for a, b in zip(contrast_alpha, contrast_beta):
                    if verbose: print("Contrast stretching augmentation called with alpha %.2f, beta %.2f." % (a, b))
                    augmented = augmented.append(self.__constrast_stretching(train, alpha = a, beta = b, verbose = verbose)).reset_index().drop(columns=['index'])

            #write train pickle file
            if verbose: print("Writing TRAIN AUGMENTATION pickle file to '%s'..." % __PICKLE_FILE_TRAIN_AUGMENT)
            pickle.dump(augmented, open(__PICKLE_FILE_TRAIN_AUGMENT, "wb"))
        else:
            if verbose: print("Loading TRAIN AUGMENTATION from pickle file '%s'" % __PICKLE_FILE_TRAIN_AUGMENT)
            augmented = pickle.load(open(__PICKLE_FILE_TRAIN_AUGMENT, "rb"))
        
        return augmented

    ##############################################
    ### Label Normalization
    ##############################################
    
    def Normalize_Labels(self, arr, verbose = False):
        """Normalizes a 1D array passed in with an assumed desired range of [-1,1] and natural range of [0,96]"""

        normalized = self.__normalize_1Darray(arr = arr, range_min = 0.,
            range_max = 96., t_min = -1., t_max = 1., verbose = verbose)
        
        return normalized

    ##############################################
    ### Label Un-Normalization
    ##############################################
    
    def UnNormalize_Labels(self, arr, verbose = False):
        """Normalizes a 1D array passed in with an assumed t_max of 1 and assumed initial range of 0-96"""

        unnormalized = self.__unnormalize_1Darray(arr = arr, t_min = -1., 
            midpoint = 48., verbose = verbose)

        return unnormalized

    ##############################################
    ### Prepare TRAIN dataframe for training (X,Y)
    ##############################################

    def PrepareTrain(self, train, feature_name, normalize = True, verbose = False):
        """Prepares the input TRAIN data into the proper shape, conditions the labels, and splits for X,Y"""

        if verbose: print("Preparing X,Y datasets for Training of '%s' model..." % feature_name)

        if feature_name == "ALL_FEATURES":
            cols = train.columns
            # for the "ALL" features, we have a lot of missing labels so we need ot address hits
            #train = train.fillna(method = 'ffill')  # maybe use the average position instead?  evaluate that later.

            # use the average values instead for the "all features" models
            #for c in [col for col in train.columns if not col == 'image']:
            #    train[c] = train[c].fillna(np.nanmean(train[c].values))
        else:
            cols = [c for c in train.columns if c.startswith(feature_name) or c == 'image']
        
        # drop the nulls from our subset of the keypoint we're training on
        subset = train[cols].copy().dropna(axis = 'index', how = 'any')
        subset.image = subset.image.map(lambda x: np.array(x).reshape(96,96,1))
        # get X into the proper shape (###, 96, 96, 1)
        X = []
        for i, r in subset.iterrows():
            X.append(r.image)
        X = np.array(X)
        
        # condition the labels through normalization prior to training
        Y = subset.drop(columns=['image']).values
        if normalize:
            for i in range(Y.shape[1]):
                Y[:,i] = self.Normalize_Labels(Y[:,i])

        return X, Y

    ##############################################
    ### Prepare TEST dataframe for inferencing
    ##############################################

    def PrepareTest(self, test, ids, feature_name, verbose = False):
        """Prepares the input TEST data into the proper shape for inferencing"""

        if verbose: print("Preparing TEST for inferencing of '%s'..." % feature_name)

        # CDB 3/18 : removed feature-specific predictions; just predict all features for all keypoints
        #if feature_name == "ALL_FEATURES":
        #    uq_id = ids.image_id.unique()
        #else:
        #    uq_id = ids[(ids.feature_name.str.startswith(feature_name))]['image_id'].unique()
        uq_id = ids.image_id.unique()

        subset = test[(test.image_id.isin(uq_id))]
        subset.image = subset.image.map(lambda x: np.array(x).reshape(96,96,1))
        # get X into the proper shape (###, 96, 96, 1)
        X = []
        for i, r in subset.iterrows():

            #CDB test sharpening images
            ########################################################################
            #img = np.clip(np.uint8(r.image * 255.0), 0, 255)
            #img = ndimage.convolve(img, KERNEL_SHARPEN, mode = 'nearest')
            #img = rescale_intensity(image = img, out_range = (0, 255))
            #img = img / 255.0
            #X.append(img)
            ########################################################################
            
            X.append(r.image)
        X = np.array(X)
        subset.drop(columns=['image'], inplace = True)

        return X, subset