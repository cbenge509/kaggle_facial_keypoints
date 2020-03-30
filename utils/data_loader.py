import os
import pandas as pd
import numpy as np
import pickle

class DataLoader(object):

    def __init__(self, data_path, pickle_path, train_data_file, test_data_file, id_lookup_file, sample_submission_file, verbose = False):
        
        # validate that the constructor parameters were provided by caller
        if (not data_path) | (not train_data_file) | (not test_data_file) | (not id_lookup_file) | (not sample_submission_file):
            raise RuntimeError('path and filenames must be provided for train, test, id, and sample submission files.')
        
        # ensure all are string snd leading/trailing whitespace removed
        data_path = str(data_path).replace('\\', '/').strip()
        if (not data_path.endswith('/')): data_path = ''.join((data_path, '/'))

        pickle_path = str(pickle_path).replace('\\', '/').strip()
        if (not pickle_path.endswith('/')): pickle_path = ''.join((pickle_path, '/'))

        train_data_file = str(train_data_file).strip()
        test_data_file = str(test_data_file).strip()
        sample_submission_file = str(sample_submission_file).strip()

        # validate the existence of the data path
        if (not os.path.isdir(data_path)):
            raise RuntimeError("Data path specified'%s' is invalid." % data_path)

        # validate the existence of the pickle path
        if (not os.path.isdir(pickle_path)):
            raise RuntimeError("Pickle path specified'%s' is invalid." % pickle_path)
        
        self.__pickle_path = pickle_path

        # validate existence of the path and data files
        for i, j in [[train_data_file, 'Train'], [test_data_file, 'Test'], [id_lookup_file, 'ID'], [sample_submission_file, 'Sample']]:
            i = ''.join((data_path, i)).replace('\\', '/')
            if (not os.path.isfile(i)):
                raise RuntimeError("'%s' data file specified [%s] does not exist." % (j, i))
            if j == 'Train':
                self.__train_file = i
            elif j == 'Test':
                self.__test_file = i
            elif j == 'ID':
                self.__id_file = i
            else:
                self.__sample_file = i

        if verbose: print("All input file locations validated.")

    def LoadRawData(self, verbose = False, recalculate_pickle = False):
        """Returns the Kaggle Facial Keypoints Detection files as Pandas dataframes in the following order: ID file, sample submission, test, train"""
    
        # pickle file names for the raw data load
        __PICKLE_FILE_TRAIN = "".join((self.__pickle_path, "raw_train.pkl"))
        __PICKLE_FILE_TEST = "".join((self.__pickle_path, "raw_test.pkl"))
        __PICKLE_FILE_IDLOOKUP = "".join((self.__pickle_path, "raw_id_lookup.pkl"))
        __PICKLE_FILE_SAMPLE = "".join((self.__pickle_path, "raw_sample_submission.pkl"))

        # Load the raw ID_LOOKUP file
        if (not os.path.isfile(__PICKLE_FILE_IDLOOKUP)) or recalculate_pickle:
            if verbose: print("Pickle file for ID_LOOKUP not found or skipped by caller.")
            try:
                id_lookup = pd.read_csv(self.__id_file, names = ['row_id', 'image_id', 'feature_name', 'location'],
                    dtype = {'row_id':'uint16', 'image_id':'uint16', 'location':'float32'}, skiprows = 1)
            except:
                raise RuntimeError("Failed to load Kaggle Facial Keypoints Detection challenge file '%s'." % self.__id_file)
            if verbose: print("Writing ID_LOOKUP to pickle file '%s'..." % __PICKLE_FILE_IDLOOKUP)
            pickle.dump(id_lookup, open(__PICKLE_FILE_IDLOOKUP, "wb"))
        else:
            if verbose: print("Loading ID_LOOKUP from pickle file '%s'" % __PICKLE_FILE_IDLOOKUP)
            id_lookup = pickle.load(open(__PICKLE_FILE_IDLOOKUP, "rb"))


        # Load the raw SAMPLE_SUBMISSION file
        if (not os.path.isfile(__PICKLE_FILE_SAMPLE)) or recalculate_pickle:
            if verbose: print("Pickle file for SAMPLE_SUBMISSION not found or skipped by caller.")
            try:
                sample = pd.read_csv(self.__sample_file, names = ['row_id', 'location'],
                    dtype = {'row_id':'uint16', 'location':'float32'}, skiprows = 1)
            except:
                raise RuntimeError("Failed to load Kaggle Facial Keypoints Detection challenge file '%s'." % self.__sample_file)
            if verbose: print("Writing SAMPLE_SUBMISSION to pickle file '%s'..." % __PICKLE_FILE_SAMPLE)
            pickle.dump(sample, open(__PICKLE_FILE_SAMPLE, "wb"))
        else:
            if verbose: print("Loading SAMPLE_SUBMISSION from pickle file '%s'" % __PICKLE_FILE_SAMPLE)
            sample = pickle.load(open(__PICKLE_FILE_SAMPLE, "rb"))

        # Load the raw TRAIN file
        if (not os.path.isfile(__PICKLE_FILE_TRAIN)) or recalculate_pickle:
            if verbose: print("Pickle file for TRAIN not found or skipped by caller.")
            try:
                train = pd.read_csv(self.__train_file, names = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 
                                    'left_eye_outer_corner_x', 'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x', 
                                    'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 
                                    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 
                                    'mouth_left_corner_x', 'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 
                                    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y', 'image'],
                    dtype = {'left_eye_center_x':'float32', 'left_eye_center_y':'float32', 'right_eye_center_x':'float32', 'right_eye_center_y':'float32', 
                                    'left_eye_inner_corner_x':'float32', 'left_eye_inner_corner_y':'float32', 'left_eye_outer_corner_x':'float32', 'left_eye_outer_corner_y':'float32', 
                                    'right_eye_inner_corner_x':'float32', 'right_eye_inner_corner_y':'float32', 'right_eye_outer_corner_x':'float32', 'right_eye_outer_corner_y':'float32', 
                                    'left_eyebrow_inner_end_x':'float32', 'left_eyebrow_inner_end_y':'float32', 'left_eyebrow_outer_end_x':'float32', 'left_eyebrow_outer_end_y':'float32', 
                                    'right_eyebrow_inner_end_x':'float32', 'right_eyebrow_inner_end_y':'float32', 'right_eyebrow_outer_end_x':'float32', 'right_eyebrow_outer_end_y':'float32', 
                                    'nose_tip_x':'float32', 'nose_tip_y':'float32', 'mouth_left_corner_x':'float32', 'mouth_left_corner_y':'float32', 'mouth_right_corner_x':'float32', 
                                    'mouth_right_corner_y':'float32', 'mouth_center_top_lip_x':'float32', 'mouth_center_top_lip_y':'float32', 'mouth_center_bottom_lip_x':'float32', 
                                    'mouth_center_bottom_lip_y':'float32', 'image':'object'}, skiprows = 1)
                if verbose: print("\tProcessing %d images..." % train.shape[0])
                train['image'] = train["image"].map(lambda x: np.array(list(map(int, x.split(" ")))))

            except:
                raise RuntimeError("Failed to load Kaggle Facial Keypoints Detection challenge file '%s'." % self.__train_file)
            if verbose: print("Writing TRAIN to pickle file '%s'..." % __PICKLE_FILE_TRAIN)
            pickle.dump(train, open(__PICKLE_FILE_TRAIN, "wb"))
        else:
            if verbose: print("Loading TRAIN from pickle file '%s'" % __PICKLE_FILE_TRAIN)
            train = pickle.load(open(__PICKLE_FILE_TRAIN, "rb"))

        # Load the raw TEST file
        if (not os.path.isfile(__PICKLE_FILE_TEST)) or recalculate_pickle:
            if verbose: print("Pickle file for TEST not found or skipped by caller.")
            try:
                test = pd.read_csv(self.__test_file, names =  ['image_id', 'image'],
                    dtype = {'image_id':'uint16', 'image':'object'}, skiprows = 1)
                if verbose: print("\tProcessing %d images..." % test.shape[0])
                test['image'] = test["image"].map(lambda x: np.array(list(map(int, x.split(" ")))))

            except:
                raise RuntimeError("Failed to load Kaggle Facial Keypoints Detection challenge file '%s'." % self.__test_file)
            if verbose: print("Writing TEST to pickle file '%s'..." % __PICKLE_FILE_TEST)
            pickle.dump(test, open(__PICKLE_FILE_TEST, "wb"))
        else:
            if verbose: print("Loading TEST from pickle file '%s'" % __PICKLE_FILE_TEST)
            test = pickle.load(open(__PICKLE_FILE_TEST, "rb"))

        return id_lookup, sample, test, train
