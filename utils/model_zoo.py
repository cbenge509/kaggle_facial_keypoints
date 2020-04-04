###########################################################################################################
## IMPORTS
###########################################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, ZeroPadding2D, GlobalMaxPooling2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
#from keras.activations import linear, elu, tanh, relu
from keras import metrics, losses, initializers, backend
from keras.utils import multi_gpu_model
from keras.initializers import glorot_uniform, Constant, lecun_uniform

os.environ["PATH"] += os.pathsep + "C:/ProgramData/Anaconda3/GraphViz/bin/"
os.environ["PATH"] += os.pathsep + "C:/Anaconda/Graphviz2.38/bin/"

from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.list_physical_devices('GPU')
for pd_dev in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)

##from tensorflow.compat.v1.keras.backend import set_session
##config = tf.compat.v1.ConfigProto()
##config.gpu_options.per_process_gpu_memory_fraction = 0.9
##config.gpu_options.allow_growth = True
##config.log_device_placement = True
##set_session(config)

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True
#sess = tf.compat.v1.InteractiveSession(config = config)
#set_session(sess)
#backend.set_session(sess)

###########################################################################################################
## PLOTTING PALETTE
###########################################################################################################

# Create a dict object containing U.C. Berkeley official school colors for plot palette 
# reference : https://alumni.berkeley.edu/brand/color-palette

berkeley_palette = {'berkeley_blue'     : '#003262',
                    'california_gold'   : '#FDB515',
                    'metallic_gold'     : '#BC9B6A',
                    'founders_rock'     : '#2D637F',
                    'medalist'          : '#E09E19',
                    'bay_fog'           : '#C2B9A7',
                    'lawrence'          : '#00B0DA',
                    'sather_gate'       : '#B9D3B6',
                    'pacific'           : '#53626F',
                    'soybean'           : '#9DAD33',
                    'california_purple' : '#5C3160',
                    'south_hall'        : '#6C3302'}

###########################################################################################################
## CLASS CONTAINING MODEL ZOO
###########################################################################################################
class Models(object):

    def __init__(self, model_path, **kwargs):
        super(Models, self).__init__(** kwargs)

        # validate that the constructor parameters were provided by caller
        if (not model_path):
            raise RuntimeError('path to model files must be provided on initialization.')
        
        # ensure all are string snd leading/trailing whitespace removed
        model_path = str(model_path).replace('\\', '/').strip()
        if (not model_path.endswith('/')): model_path = ''.join((model_path, '/'))

        # validate the existence of the data path
        if (not os.path.isdir(model_path)):
            raise RuntimeError("Models path specified'%s' is invalid." % model_path)

        self.__models_path = model_path
        self.__GPU_count = len(tf.config.list_physical_devices('GPU'))
        self.__MIN_early_stopping = 10

    #------------------------------------------------
    # Private Methods
    #------------------------------------------------

    # plotting method for keras history arrays
    def __plot_keras_history(self, history, metric, model_name, feature_name, file_name, verbose = False):
            # Plot the performance of the model training
            fig = plt.figure(figsize=(15,8),dpi=80)
            ax = fig.add_subplot(121)

            ax.plot(history.history[metric][1:], color = berkeley_palette['founders_rock'], label = 'Train',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.plot(history.history["".join(["val_",metric])][1:], color = berkeley_palette['medalist'], label = 'Validation',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.set_title(" ".join(['Model Performance',"(" + model_name + ")"]) + "\n" + feature_name, 
                color = berkeley_palette['berkeley_blue'], fontsize = 15, fontweight = 'bold')
            ax.spines["top"].set_alpha(.0)
            ax.spines["bottom"].set_alpha(.3)
            ax.spines["right"].set_alpha(.0)
            ax.spines["left"].set_alpha(.3)
            ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0, color = berkeley_palette['berkeley_blue'])
            ax.set_ylabel(metric, fontsize = 12, horizontalalignment='right', y = 1.0, color = berkeley_palette['berkeley_blue'])
            plt.legend(loc = 'upper right')

            ax = fig.add_subplot(122)

            ax.plot(history.history['loss'][1:], color = berkeley_palette['founders_rock'], label = 'Train',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.plot(history.history["".join(["val_loss"])][1:], color = berkeley_palette['medalist'], label = 'Validation',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.set_title(" ".join(['Model Performance',"(" + model_name + ")"]) + "\n" + feature_name, 
                color = berkeley_palette['berkeley_blue'], fontsize = 15, fontweight = 'bold')
            ax.spines["top"].set_alpha(.0)
            ax.spines["bottom"].set_alpha(.3)
            ax.spines["right"].set_alpha(.0)
            ax.spines["left"].set_alpha(.3)
            ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0, color = berkeley_palette['berkeley_blue'])
            ax.set_ylabel("Loss", fontsize = 12, horizontalalignment='right', y = 1.0, color = berkeley_palette['berkeley_blue'])
            plt.legend(loc = 'upper right')

            plt.tight_layout()
            plt.savefig(file_name, dpi=300)
            if verbose: print("Training plot file saved to '%s'." % file_name)
            plt.close()

    # load Keras model files from json / h5
    def __load_keras_model(self, model_name, model_file, model_json, verbose = False):
        """Loads a Keras model from disk"""

        if not os.path.isfile(model_file):
            raise RuntimeError("Model file '%s' does not exist; exiting inferencing." % model_file)
        if not os.path.isfile(model_json):
            raise RuntimeError("Model file '%s' does not exist; exiting inferencing." % model_json)

        # load model file
        if verbose: print("Retrieving model: %s..." % model_name)
        json_file = open(model_json, "r")
        model_json_data = json_file.read()
        json_file.close()
        model = model_from_json(model_json_data)
        model.load_weights(model_file)
        
        return model

    # Performs standard scaling on a 4D image
    def __4d_Scaler(self, arr, ss, fit = False, verbose = False):
        """Performs standard scaling of the 4D array with the 'ss' model provided by caller"""
        
        #Unwinds a (instances, rows, columns, layers) array to 2D for standard scaling
        num_instances, num_rows, num_columns, num_layers = arr.shape
        arr_copy = np.reshape(arr, (-1, num_columns))
        
        # fit the standard scaler
        if fit:
            if verbose: print("Fitting SCALER and transforming...")
            arr_copy = ss.fit_transform(arr_copy)
        else:
            if verbose: print("Transforming SCALER only...")
            arr_copy = ss.transform(arr_copy)
        
        arr = np.reshape(arr_copy, (num_instances, num_rows, num_columns, num_layers))

        return arr

    # resnet identity block builder
    def __identity_block(self, model, kernel_size, filters, stage, block):
        """modularized identity block for resnet"""
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(model)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = add([x, model])
        x = Activation('relu')(x)
        return x

    # resnet conv block builder
    def __conv_block(self, model, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv block builder for resnet"""
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(model)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x =Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                                kernel_initializer='he_normal',
                                name=conv_name_base + '1')(model)
        shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        
        return x

    # create a layerable inception module
    def __inception_module(self, model, filters_1x1, filters_3x3_reduce, filters_3x3, 
            filters_5x5_reduce, filters_5x5, filters_pool_proj, kernel_init, bias_init, name = None):
        """modularized inception block for layering"""

        # Connection Layer 1 (1x1)
        conv_1x1 = Convolution2D(filters_1x1, (1, 1), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)
        
        # Connection Layer 2 (3x3)
        conv_3x3 = Convolution2D(filters_3x3_reduce, (1, 1), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)
        conv_3x3 = Convolution2D(filters_3x3, (3, 3), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (conv_3x3)

        # Connection Layer 3 (5x5)
        conv_5x5 = Convolution2D(filters_5x5_reduce, (1, 1), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)
        conv_5x5 = Convolution2D(filters_5x5, (5, 5), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (conv_5x5)

        # Connection Layer 4 (pool)
        pool_proj = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same') (model)
        pool_proj = Convolution2D(filters_pool_proj, (1, 1), padding = 'same', activation = 'relu', 
            kernel_initializer = kernel_init, bias_initializer = bias_init) (pool_proj)

        # Concatenation layer
        output = concatenate(inputs = [conv_1x1, conv_3x3, conv_5x5, pool_proj], axis = 3, name = name)
        
        return output     

    # return an InceptionV3 output tensor after applying Conv2D and BatchNormalization
    def __conv2d_bn(self, x, filters, num_row, num_col, padding = 'same', strides = (1, 1), name = None):

        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        
        bn_axis = 3
        
        x = Convolution2D(filters, (num_row, num_col), strides = strides,
            padding = padding, use_bias = False, name = conv_name) (x)
        
        x = BatchNormalization(axis = bn_axis, scale = False, name = bn_name) (x)
        x = ReLU(name = name) (x)

        return x


    ######################################################
    ######################################################
    ######################################################
    ### KERAS MODEL ZOO
    ######################################################
    ######################################################
    ######################################################

    #------------------------------------------------
    # NaimishNet Model
    # ref: https://arxiv.org/abs/1710.00977
    #------------------------------------------------

    def get_keras_naimishnet(self, X, Y, batch_size, epoch_count, X_val = None, Y_val = None, val_split = 0.1, shuffle = True, 
        feature_name = "unknown", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - NaimishNet"
        __MODEL_FNAME_PREFIX = "KERAS_NAIMISHNET/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, "_model_plot.png"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, "_", feature_name, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']
            #ke = initializers.lecun_uniform(seed = 42)
            ke = 'glorot_uniform'

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):

                l1 = Input((96, 96, 1))

                l2 = Convolution2D(32, (4, 4), kernel_initializer = ke, padding = 'valid', activation = 'elu') (l1)
                #l3 = ELU() (l2)
                l3 = MaxPooling2D(pool_size=(2,2), strides = (2,2)) (l2)
                l4 = Dropout(rate = 0.1) (l3)

                l5 = Convolution2D(64, (3, 3), kernel_initializer = ke, padding = 'valid', activation = 'elu') (l4)
                #l7 = ELU() (l6)
                l6 = MaxPooling2D(pool_size=(2,2), strides = (2,2)) (l5)
                l7 = Dropout(rate = 0.2) (l6)

                l8 = Convolution2D(128, (2, 2), kernel_initializer = ke, padding = 'valid', activation = 'elu') (l7)
                #l11 = ELU() (l10)
                l9 = MaxPooling2D(pool_size=(2,2), strides = (2,2)) (l8)
                l10 = Dropout(rate = 0.3) (l9)

                l11 = Convolution2D(256, (1, 1), kernel_initializer = ke, padding = 'valid', activation = 'elu') (l10)
                #l15 = ELU() (l14)
                l12 = MaxPooling2D(pool_size=(2,2), strides = (2,2)) (l11)
                l13 = Dropout(rate = 0.4) (l12)

                l14 = Flatten() (l13)
                l15 = Dense(1000, activation = 'elu') (l14)
                #l20 = ELU() (l19)
                l16 = Dropout(rate = 0.5) (l15)

                #l22 = Dense(1000) (l21)
                #l23 = linear(l22)

                l17 = Dense(1000, activation = 'linear') (l16)
                l18 = Dropout(rate = 0.6) (l17)

                l19 = Dense(2) (l18)

                model = Model(inputs = [l1], outputs = [l19])
                model.compile(optimizer = act, loss = lss, metrics = mtrc)
            
            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', model_name = __MODEL_NAME, 
                feature_name = feature_name, file_name = __history_plot_file, verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)

        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            modparallel_modelel = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_naimishnet(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - NaimishNet"
        __MODEL_FNAME_PREFIX = "KERAS_NAIMISHNET/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, "NaimishNet_", feature_name, __MODEL_SUFFIX, ".json"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % 
                (__model_file_name, __model_json_file))
        
        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

    #------------------------------------------------
    # Kaggle1 Model 
    # Inspired by: https://www.kaggle.com/balraj98/data-augmentation-for-facial-keypoint-detection
    #------------------------------------------------

    def get_keras_kaggle1(self, X, Y, batch_size, epoch_count, val_split = 0.05, X_val = None, Y_val = None, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle1"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE1/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])
        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            #act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            act = 'adam'
            #lss = losses.mean_squared_error
            lss = 'mean_squared_error'
            #mtrc = [metrics.RootMeanSquaredError()]
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                model = Sequential()

                # Input dimensions: (None, 96, 96, 1)
                model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 96, 96, 32)
                model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # CDB: 3/5 DROPOUT ADDED
                model.add(Dropout(0.2))

                # Input dimensions: (None, 48, 48, 32)
                model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 48, 48, 64)
                model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # CDB: 3/5 DROPOUT ADDED
                model.add(Dropout(0.25))

                # Input dimensions: (None, 24, 24, 64)
                model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 24, 24, 96)
                model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # CDB: 3/5 DROPOUT ADDED
                model.add(Dropout(0.15))

                # Input dimensions: (None, 12, 12, 96)
                model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 12, 12, 128)
                model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # CDB: 3/5 DROPOUT ADDED
                model.add(Dropout(0.3))

                # Input dimensions: (None, 6, 6, 128)
                model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 6, 6, 256)
                model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                # CDB: 3/5 DROPOUT ADDED
                model.add(Dropout(0.2))

                # Input dimensions: (None, 3, 3, 256)
                model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                # Input dimensions: (None, 3, 3, 512)
                model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())

                # Input dimensions: (None, 3, 3, 512)
                model.add(Flatten())
                model.add(Dense(512,activation='relu'))
                
                # CDB DROPOUT INCREASED FROM 0.1 to 0.2
                model.add(Dropout(0.15))
                if full:
                    model.add(Dense(30))
                else:
                    model.add(Dense(8))

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)


            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', #metric = 'root_mean_squared_error', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)


        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_kaggle1(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle1"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE1/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):## or (not os.path.isfile(__scaler_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % ##'%s'\n" % 
                (__model_file_name, __model_json_file))##, __scaler_file))
        
        # Load the training scaler for this model
        ##if verbose: print("Loading SCALER for '%s' and zero-centering X." % feature_name)
        ##scaler = pickle.load(open(__scaler_file, "rb"))
        ##X = self.__4d_Scaler(arr = X, ss = scaler, fit = False, verbose = verbose)

        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

    #-------------------------------------------------------------
    # LeNet5 Model
    # Inspired by: Google's LeNet5 for MNSIST - Modified
    #-------------------------------------------------------------

    def get_keras_lenet5(self, X, Y, batch_size, epoch_count, X_val = None, Y_val = None, val_split = 0.1, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - LeNet5"
        __MODEL_FNAME_PREFIX = "KERAS_LENET5/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            #if (X_val is None) or (Y_val is None):
            #    if verbose: print("No validation set specified; creating a split based on %.2f val_split parameter." % val_split)
            #    X, Y, X_val, Y_val = train_test_split(X, Y, test_size = val_split, random_state = 42)

            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)
            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                model = Sequential()

                model.add(Convolution2D(filters = 6, kernel_size = (3, 3), input_shape = (96, 96, 1)))
                model.add(ReLU())
                # CDB: 3/5 added Batch Normalization
                #model.add(BatchNormalization())
                model.add(AveragePooling2D())
                #model.add(Dropout(0.2))

                model.add(Convolution2D(filters = 16, kernel_size = (3, 3)))
                model.add(ReLU())
                # CDB: 3/5 added Batch Normalization
                #model.add(BatchNormalization())
                model.add(AveragePooling2D())
                #model.add(Dropout(0.2))

                model.add(Flatten())
                model.add(Dense(512))
                model.add(ReLU())
                #model.add(Dropout(0.1))

                model.add(Dense(256))
                model.add(ReLU())
                #model.add(Dropout(0.2))

                if full:
                    model.add(Dense(30))
                else:
                    model.add(Dense(8))

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
            else:
                parallel_model = model

            parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', model_name = __MODEL_NAME, 
                feature_name = feature_name, file_name = __history_plot_file, verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            
            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)

            if verbose: print("Model JSON, history, and parameters file saved.")

        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_lenet5(self, X, feature_name = "ALL_FEATURES", full = True, verbose = False):

        __MODEL_NAME = "Keras - LeNet5"
        __MODEL_FNAME_PREFIX = "KERAS_LENET5/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % 
                (__model_file_name, __model_json_file))
        
        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

    #-------------------------------------------------------------
    # Inception V1
    # Inspired by : https://arxiv.org/abs/1409.4842
    #-------------------------------------------------------------

    def get_keras_inception(self, X, Y, batch_size, epoch_count, val_split = 0.1, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, X_val = None, Y_val = None, full = True, verbose = False):

        __MODEL_NAME = "Keras - Inception"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTION/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_MAIN_name = "".join([nested_dir, "inception_MAIN_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_file_AUX1_name = "".join([nested_dir, "inception_AUX1_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_file_AUX2_name = "".join([nested_dir, "inception_AUX2_", feature_name, __MODEL_SUFFIX, ".h5"])

        __model_json_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, ".json"])
        __model_architecture_plot_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_plot.png"])
        __history_params_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file_main = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_main_output_mse_plot.png"])
        __history_plot_file_auxilliary1 = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_auxilliary_output_1_mse_plot.png"])
        __history_plot_file_auxilliary2 = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_auxilliary_output_2_mse_plot.png"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_file_AUX1_name)) or (not os.path.isfile(__model_file_AUX2_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % __MODEL_NAME)

            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']
            
            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp_main = ModelCheckpoint(filepath = __model_file_MAIN_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_main_output_mae')
            cp_aux1 = ModelCheckpoint(filepath = __model_file_AUX1_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_auxilliary_output_1_mae')
            cp_aux2 = ModelCheckpoint(filepath = __model_file_AUX2_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_auxilliary_output_2_mae')

            kernel_init = glorot_uniform()
            bias_init = Constant(value = 0.2)

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                
                # Input image shape (H, W, C)
                input_img = Input(shape=(96, 96, 1))

                # Top Layer (Begin MODEL)
                model = Convolution2D(filters = 64, kernel_size = (7, 7), padding = 'same', strides = (2, 2),
                    activation = 'relu', name = 'conv_1_7x7/2', kernel_initializer = kernel_init, 
                    bias_initializer = bias_init) (input_img)
                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_1_3x3/2') (model)
                model = Convolution2D(64, (1, 1), padding = 'same', strides = (1, 1), activation = 'relu', name = 'conv_2a_3x3/1') (model)
                model = Convolution2D(192, (3, 3), padding = 'same', strides = (1, 1), activation = 'relu', name = 'conv_2b_3x3/1') (model)
                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_2_3x3/2') (model)

                # Inception Module
                model = self.__inception_module(model, 
                    filters_1x1 = 64, 
                    filters_3x3_reduce = 96, 
                    filters_3x3 = 128, 
                    filters_5x5_reduce = 16, 
                    filters_5x5 = 32, 
                    filters_pool_proj = 32, 
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name = 'inception_3a')

                # Inception Module
                model = self.__inception_module(model, 
                    filters_1x1 = 128, 
                    filters_3x3_reduce = 128, 
                    filters_3x3 = 192, 
                    filters_5x5_reduce = 32, 
                    filters_5x5 = 96, 
                    filters_pool_proj = 64, 
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name = 'inception_3b')

                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name= 'max_pool_3_3x3/2') (model)

                # Inception Module
                model = self.__inception_module(model, 
                    filters_1x1 = 192, 
                    filters_3x3_reduce = 96, 
                    filters_3x3 = 208, 
                    filters_5x5_reduce = 16, 
                    filters_5x5 = 48, 
                    filters_pool_proj = 64, 
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name = 'inception_4a')
                
                # CDB 3/5 DROPOUT ADDED
                model = Dropout(0.2) (model)

                # Begin MODEL1 (auxillary output)
                model1 = AveragePooling2D((5, 5), padding = 'same', strides = 3, name= 'avg_pool_4_5x5/2') (model)

                model1 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model1)
                model1 = Flatten() (model1)
                model1 = Dense(1024, activation = 'relu') (model1)
                model1 = Dropout(0.3) (model1)
                if full:
                    model1 = Dense(30, name = 'auxilliary_output_1') (model1)
                else:
                    model1 = Dense(8, name = 'auxilliary_output_1') (model1)

                # Resume MODEL w/ Inception
                model = self.__inception_module(model,
                    filters_1x1 = 160,
                    filters_3x3_reduce = 112,
                    filters_3x3 = 224,
                    filters_5x5_reduce = 24,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_4b')

                model = self.__inception_module(model,
                    filters_1x1 = 128,
                    filters_3x3_reduce = 128,
                    filters_3x3 = 256,
                    filters_5x5_reduce = 24,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_4c')

                model = self.__inception_module(model,
                    filters_1x1 = 112,
                    filters_3x3_reduce = 144,
                    filters_3x3 = 288,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_4d')
                
                # CDB : 3/5 DROPOUT ADDED
                model = Dropout(0.2) (model)

                # Begin MODEL2 (auxilliary output)
                model2 = AveragePooling2D((5, 5), strides = 3) (model)
                model2 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model2)
                model2 = Flatten() (model2)
                model2 = Dense(1024, activation = 'relu') (model2)
                model2 = Dropout(0.3) (model2)
                if full:
                    model2 = Dense(30, name = 'auxilliary_output_2') (model2)
                else:
                    model2 = Dense(8, name = 'auxilliary_output_2') (model2)

                # Resume MODEL w/ Inception
                model = self.__inception_module(model,
                    filters_1x1 = 256,
                    filters_3x3_reduce = 160,
                    filters_3x3 = 320,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_4e')

                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_4_3x3/2') (model)

                model = self.__inception_module(model,
                    filters_1x1 = 256,
                    filters_3x3_reduce = 160,
                    filters_3x3 = 320,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_5a')

                model = self.__inception_module(model,
                    filters_1x1 = 384,
                    filters_3x3_reduce = 192,
                    filters_3x3 = 384,
                    filters_5x5_reduce = 48,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init, 
                    bias_init = bias_init, 
                    name='inception_5b')

                model = GlobalAveragePooling2D(name = 'avg_pool_5_3x3/1') (model)
                model = Dropout(0.3) (model)

                # Output Layer (Main)
                if full:
                    model = Dense(30, name = 'main_output') (model)
                else:
                    model = Dense(8, name = 'main_output') (model)
                
                model = Model(input_img, [model, model1, model2], name = 'Inception')

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, [Y, Y, Y], validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main, cp_aux1, cp_aux2], verbose = verbose)
            else:
                history = parallel_model.fit(X, [Y, Y, Y], validation_data = (X_val, [Y_val, Y_val, Y_val]), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main, cp_aux1, cp_aux2], verbose = verbose)

            # print and/or save a performance plot
            for m, f in zip(['main_output_mse', 'auxilliary_output_1_mse', 'auxilliary_output_2_mse'], 
                [__history_plot_file_main, __history_plot_file_auxilliary1, __history_plot_file_auxilliary2]):
                try:
                    self.__plot_keras_history(history = history, metric = m, model_name = __MODEL_NAME, 
                        feature_name = feature_name, file_name = f, verbose = False)
                except:
                    print("error during history plot generation; skipped.")
                    pass
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            
            # save a plot of the model architecture
            try:
                plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                    show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)
            except:
                print("error during model plot generation; skiopped.")
                pass

            if verbose: print("Model JSON, history, and parameters file saved.")

        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)

        if verbose: print("Loading pickle file for '%s' MODEL (MAIN) from file '%s'" % (__MODEL_NAME, __model_file_MAIN_name))
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)

        if verbose: print("Loading pickle file for '%s' MODEL (AUX1) from file '%s'" % (__MODEL_NAME, __model_file_AUX1_name))
        aux1_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX1_name, __model_json_file, verbose = verbose)

        if verbose: print("Loading pickle file for '%s' MODEL (AUX2) from file '%s'" % (__MODEL_NAME, __model_file_AUX2_name))
        aux2_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX2_name, __model_json_file, verbose = verbose)

        return main_model, aux1_model, aux2_model, hist_params, hist

    # inferencing
    def predict_keras_inception(self, X, feature_name = "ALL_FEATURES", full = True, verbose = False):

        __MODEL_NAME = "Keras - Inception"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTION/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_MAIN_name = "".join([nested_dir, "inception_MAIN_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_file_AUX1_name = "".join([nested_dir, "inception_AUX1_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_file_AUX2_name = "".join([nested_dir, "inception_AUX2_", feature_name, __MODEL_SUFFIX, ".h5"])

        __model_json_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, ".json"])


        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_file_AUX1_name)) or (not os.path.isfile(__model_file_AUX2_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n'%s'\n'%s'\n\n" % 
                (__model_file_MAIN_name, __model_file_AUX1_name, __model_file_AUX2_name, __model_json_file))
        
        # load the Keras model for the specified feature
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)
        aux1_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX1_name, __model_json_file, verbose = verbose)
        aux2_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX2_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for 'MAIN' model file..." % len(X))
        Y_main = main_model.predict(X, verbose = verbose)
        Y_main_columns = [node.op.name for node in main_model.outputs]

        if verbose: print("Predicting %d (x,y) coordinates for 'AUX1' model file..." % len(X))
        Y_aux1 = aux1_model.predict(X, verbose = verbose)
        Y_aux1_columns = [node.op.name for node in aux1_model.outputs]

        if verbose: print("Predicting %d (x,y) coordinates for 'AUX2' model file..." % len(X))
        Y_aux2 = aux2_model.predict(X, verbose = verbose)
        Y_aux2_columns = [node.op.name for node in aux2_model.outputs]

        if verbose: print("Predictions completed!")

        return Y_main, Y_aux1, Y_aux2, Y_main_columns, Y_aux1_columns, Y_aux2_columns

    #------------------------------------------------
    # ConvNet5 Simple Model
    #------------------------------------------------

    def get_keras_convnet5(self, X, Y, batch_size, epoch_count, val_split = 0.1, X_val = None, Y_val = None, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - ConvNet5"
        __MODEL_FNAME_PREFIX = "KERAS_CONVNET5/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])


        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):

                model = Sequential(name = 'ConvNet5')

                # Input dimensions: (None, 96, 96, 1)
                model.add(Convolution2D(16, (3,3), padding = 'same', activation = 'relu', input_shape=(96,96,1)))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Dropout(0.1))

                model.add(Convolution2D(32, (3,3), padding = 'same', activation = 'relu'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Dropout(0.2))

                model.add(Convolution2D(64, (3,3), padding = 'same', activation = 'relu'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Dropout(0.1))

                model.add(Convolution2D(128, (3,3), padding = 'same', activation = 'relu'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Dropout(0.25))

                model.add(Convolution2D(256, (3,3), padding = 'same', activation = 'relu'))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Dropout(0.15))

                model.add(Flatten())
                model.add(Dense(1024, activation = 'relu'))
                model.add(Dropout(0.1))

                model.add(Dense(512, activation = 'relu'))
                model.add(Dropout(0.1))

                if full:
                    model.add(Dense(30))
                else:
                    model.add(Dense(8))

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)


            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)

        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_convnet5(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - ConvNet5"
        __MODEL_FNAME_PREFIX = "KERAS_CONVNET5/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % 
                (__model_file_name, __model_json_file))
        
        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

    #-------------------------------------------------------------
    # Inception V3
    # Inspired by : http://arxiv.org/abs/1512.00567
    #-------------------------------------------------------------

    def get_keras_inceptionv3(self, X, Y, batch_size, epoch_count, val_split = 0.1, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, X_val = None, Y_val = None, full = True, verbose = False):

        __MODEL_NAME = "Keras - Inceptionv3"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTIONV3/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_MAIN_name = "".join([nested_dir, "inception_MAIN_", feature_name, __MODEL_SUFFIX, ".h5"])

        __model_json_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, ".json"])
        __model_architecture_plot_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_plot.png"])
        __history_params_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file_main = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, "_main_output_mse_plot.png"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % __MODEL_NAME)

            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']
            
            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp_main = ModelCheckpoint(filepath = __model_file_MAIN_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            kernel_init = glorot_uniform()
            bias_init = Constant(value = 0.2)

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                
                # Input image shape (H, W, C)
                input_img = Input(shape = (96, 96, 1))

                # Begin Inception V3
                x = self.__conv2d_bn(x = input_img, filters = 32, num_row = 3, num_col = 3, strides = (2, 2), padding = 'valid')
                x = self.__conv2d_bn(x = x, filters = 32, num_row = 3, num_col = 3, strides = (1, 1), padding = 'valid')
                x = self.__conv2d_bn(x = x, filters = 64, num_row = 3, num_col = 3, strides = (1, 1), padding = 'same')
                x = MaxPooling2D((3, 3), strides = (2, 2)) (x)

                x = self.__conv2d_bn(x = x, filters = 80, num_row = 1, num_col = 1, strides = (1, 1), padding = 'valid')
                x = self.__conv2d_bn(x = x, filters = 192, num_row = 3, num_col = 3, strides = (1, 1), padding = 'valid')
                x = MaxPooling2D((3, 3), strides = (2, 2)) (x)

                branch1x1 = self.__conv2d_bn(x = x, filters = 64, num_row = 1, num_col = 1, strides = (1, 1), padding = 'same')
                
                branch5x5 = self.__conv2d_bn(x = x, filters = 48, num_row = 1, num_col = 1, strides = (1, 1), padding = 'same')
                branch5x5 = self.__conv2d_bn(x = branch5x5, filters = 64, num_row = 5, num_col = 5, strides = (1, 1), padding = 'same')

                branch3x3dbl = self.__conv2d_bn(x = x, filters = 64, num_row = 1, num_col = 1, strides = (1, 1), padding = 'same')
                branch3x3dbl = self.__conv2d_bn(x = branch3x3dbl, filters = 96, num_row = 3, num_col = 3, strides = (1, 1), padding = 'same')
                branch3x3dbl = self.__conv2d_bn(x = branch3x3dbl, filters = 96, num_row = 3, num_col = 3, strides = (1, 1), padding = 'same')

                branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                branch_pool = self.__conv2d_bn(x = branch_pool, filters = 32, num_row = 1, num_col = 1, strides = (1, 1), padding = 'same')
                x = concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis = 3, name = 'mixed0')

                branch1x1 = self.__conv2d_bn(x, 64, 1, 1)

                branch5x5 = self.__conv2d_bn(x, 48, 1, 1)
                branch5x5 = self.__conv2d_bn(branch5x5, 64, 5, 5)

                branch3x3dbl = self.__conv2d_bn(x, 64, 1, 1)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3)

                branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                branch_pool = self.__conv2d_bn(branch_pool, 64, 1, 1)
                x = concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis = 3, name = 'mixed1')

                branch1x1 = self.__conv2d_bn(x, 64, 1, 1)

                branch5x5 = self.__conv2d_bn(x, 48, 1, 1)
                branch5x5 = self.__conv2d_bn(branch5x5, 64, 5, 5)

                branch3x3dbl = self.__conv2d_bn(x, 64, 1, 1)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3)

                branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                branch_pool = self.__conv2d_bn(branch_pool, 64, 1, 1)
                x = concatenate( [branch1x1, branch5x5, branch3x3dbl, branch_pool], axis = 3, name = 'mixed2')

                branch3x3 = self.__conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

                branch3x3dbl = self.__conv2d_bn(x, 64, 1, 1)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3)
                branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

                branch_pool = MaxPooling2D((3, 3), strides=(2, 2)) (x)
                x = concatenate( [branch3x3, branch3x3dbl, branch_pool], axis = 3, name = 'mixed3')

                branch1x1 = self.__conv2d_bn(x, 192, 1, 1)

                branch7x7 = self.__conv2d_bn(x, 128, 1, 1)
                branch7x7 = self.__conv2d_bn(branch7x7, 128, 1, 7)
                branch7x7 = self.__conv2d_bn(branch7x7, 192, 7, 1)

                branch7x7dbl = self.__conv2d_bn(x, 128, 1, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 128, 7, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 128, 1, 7)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 128, 7, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 1, 7)

                branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                branch_pool = self.__conv2d_bn(branch_pool, 192, 1, 1)
                x = concatenate( [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis = 3, name = 'mixed4')

                for i in range(2):
                    branch1x1 = self.__conv2d_bn(x, 192, 1, 1)

                    branch7x7 = self.__conv2d_bn(x, 160, 1, 1)
                    branch7x7 = self.__conv2d_bn(branch7x7, 160, 1, 7)
                    branch7x7 = self.__conv2d_bn(branch7x7, 192, 7, 1)

                    branch7x7dbl = self.__conv2d_bn(x, 160, 1, 1)
                    branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 160, 7, 1)
                    branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 160, 1, 7)
                    branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 160, 7, 1)
                    branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 1, 7)

                    branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                    branch_pool = self.__conv2d_bn(branch_pool, 192, 1, 1)
                    x = concatenate( [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis = 3, name = 'mixed' + str(5 + i))

                branch1x1 = self.__conv2d_bn(x, 192, 1, 1)

                branch7x7 = self.__conv2d_bn(x, 192, 1, 1)
                branch7x7 = self.__conv2d_bn(branch7x7, 192, 1, 7)
                branch7x7 = self.__conv2d_bn(branch7x7, 192, 7, 1)

                branch7x7dbl = self.__conv2d_bn(x, 192, 1, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 7, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 1, 7)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 7, 1)
                branch7x7dbl = self.__conv2d_bn(branch7x7dbl, 192, 1, 7)

                branch_pool = AveragePooling2D((3, 3), strides = (1, 1), padding = 'same') (x)
                branch_pool = self.__conv2d_bn(branch_pool, 192, 1, 1)
                x = concatenate( [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis = 3, name = 'mixed7')

                branch3x3 = self.__conv2d_bn(x, 192, 1, 1)
                branch3x3 = self.__conv2d_bn(branch3x3, 320, 3, 3,strides=(2, 2), padding='valid')

                branch7x7x3 = self.__conv2d_bn(x, 192, 1, 1)
                branch7x7x3 = self.__conv2d_bn(branch7x7x3, 192, 1, 7)
                branch7x7x3 = self.__conv2d_bn(branch7x7x3, 192, 7, 1)
                branch7x7x3 = self.__conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

                branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
                x = concatenate( [branch3x3, branch7x7x3, branch_pool], axis = 3, name = 'mixed8')

                for i in range(2):
                    branch1x1 = self.__conv2d_bn(x, 320, 1, 1)

                    branch3x3 = self.__conv2d_bn(x, 384, 1, 1)
                    branch3x3_1 = self.__conv2d_bn(branch3x3, 384, 1, 3)
                    branch3x3_2 = self.__conv2d_bn(branch3x3, 384, 3, 1)
                    branch3x3 = concatenate( [branch3x3_1, branch3x3_2], axis = 3, name = 'mixed9_' + str(i))

                    branch3x3dbl = self.__conv2d_bn(x, 448, 1, 1)
                    branch3x3dbl = self.__conv2d_bn(branch3x3dbl, 384, 3, 3)
                    branch3x3dbl_1 = self.__conv2d_bn(branch3x3dbl, 384, 1, 3)
                    branch3x3dbl_2 = self.__conv2d_bn(branch3x3dbl, 384, 3, 1)
                    branch3x3dbl = concatenate( [branch3x3dbl_1, branch3x3dbl_2], axis = 3)

                    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
                    branch_pool = self.__conv2d_bn(branch_pool, 192, 1, 1)
                    x = concatenate( [branch1x1, branch3x3, branch3x3dbl, branch_pool], axis = 3, name = 'mixed' + str(9 + i))

                x = GlobalAveragePooling2D(name = 'avg_pool') (x)
                x = Dropout(0.3) (x)
                if full:
                    x = Dense(30) (x)
                else:
                    x = Dense(8) (x)
                model = Model(input_img, x, name = 'InceptionV3')

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file_main,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            
            # save a plot of the model architecture
            try:
                plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                    show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)
            except:
                print("error during model plot generation; skiopped.")
                pass

            if verbose: print("Model JSON, history, and parameters file saved.")

        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)

        if verbose: print("Loading pickle file for '%s' MODEL (MAIN) from file '%s'" % (__MODEL_NAME, __model_file_MAIN_name))
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)

        return main_model, hist_params, hist

    # inferencing
    def predict_keras_inceptionv3(self, X, feature_name = "ALL_FEATURES", full = True, verbose = False):

        __MODEL_NAME = "Keras - InceptionV3"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTIONV3/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_MAIN_name = "".join([nested_dir, "inception_MAIN_", feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, "inception_", feature_name, __MODEL_SUFFIX, ".json"])


        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n%s'\n'%s'\n\n" % 
                (__model_file_MAIN_name, __model_json_file))
        
        # load the Keras model for the specified feature
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for 'MAIN' model file..." % len(X))
        Y_main = main_model.predict(X, verbose = verbose)
        Y_main_columns = [node.op.name for node in main_model.outputs]

        if verbose: print("Predictions completed!")

        return Y_main, Y_main_columns

    #------------------------------------------------
    # Kaggle2 Model 
    #------------------------------------------------

    def get_keras_kaggle2(self, X, Y, batch_size, epoch_count, val_split = 0.05, X_val = None, Y_val = None, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle2"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE2/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])
        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            #act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            act = 'adam'
            #lss = losses.mean_squared_error
            lss = 'mean_squared_error'
            #mtrc = [metrics.RootMeanSquaredError()]
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                model = Sequential()

                # Input dimensions: (None, 96, 96, 1)
                model.add(Convolution2D(32, (3,3), padding='valid', use_bias = True, input_shape = (96,96,1)))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(64, (3,3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(128, (3,3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(LocallyConnected2D(32, (3, 3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(GlobalAveragePooling2D())

                # Input dimensions: (None, 3, 3, 512)
                #model.add(Flatten())
                model.add(Dense(512,activation='relu'))
                
                # CDB DROPOUT INCREASED FROM 0.1 to 0.2
                model.add(Dropout(0.15))
                if full:
                    model.add(Dense(30))
                else:
                    model.add(Dense(8))

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)


            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', #metric = 'root_mean_squared_error', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)


        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_kaggle2(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle2"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE2/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):## or (not os.path.isfile(__scaler_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % ##'%s'\n" % 
                (__model_file_name, __model_json_file))##, __scaler_file))
        
        # Load the training scaler for this model
        ##if verbose: print("Loading SCALER for '%s' and zero-centering X." % feature_name)
        ##scaler = pickle.load(open(__scaler_file, "rb"))
        ##X = self.__4d_Scaler(arr = X, ss = scaler, fit = False, verbose = verbose)

        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y
    #------------------------------------------------
    # Kaggle2 Model 
    #------------------------------------------------

    def get_keras_kaggle2(self, X, Y, batch_size, epoch_count, val_split = 0.05, X_val = None, Y_val = None, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle2"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE2/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])
        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            #act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            act = 'adam'
            #lss = losses.mean_squared_error
            lss = 'mean_squared_error'
            #mtrc = [metrics.RootMeanSquaredError()]
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):
                model = Sequential()

                # Input dimensions: (None, 96, 96, 1)
                model.add(Convolution2D(32, (3,3), padding='valid', use_bias = True, input_shape = (96,96,1)))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(64, (3,3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(128, (3,3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(LocallyConnected2D(32, (3, 3), padding = 'valid', use_bias = True))
                model.add(LeakyReLU(alpha = 0.1))
                model.add(BatchNormalization())
                model.add(GlobalAveragePooling2D())

                # Input dimensions: (None, 3, 3, 512)
                #model.add(Flatten())
                model.add(Dense(512,activation='relu'))
                
                # CDB DROPOUT INCREASED FROM 0.1 to 0.2
                model.add(Dropout(0.15))
                if full:
                    model.add(Dense(30))
                else:
                    model.add(Dense(8))

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)


            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', #metric = 'root_mean_squared_error', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)


        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_kaggle2(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - Kaggle2"
        __MODEL_FNAME_PREFIX = "KERAS_KAGGLE2/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):## or (not os.path.isfile(__scaler_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % ##'%s'\n" % 
                (__model_file_name, __model_json_file))##, __scaler_file))
        
        # Load the training scaler for this model
        ##if verbose: print("Loading SCALER for '%s' and zero-centering X." % feature_name)
        ##scaler = pickle.load(open(__scaler_file, "rb"))
        ##X = self.__4d_Scaler(arr = X, ss = scaler, fit = False, verbose = verbose)

        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

    #------------------------------------------------
    # ResNet50
    # Inspired by: https://arxiv.org/abs/1512.03385
    #------------------------------------------------

    def get_keras_resnet50(self, X, Y, batch_size, epoch_count, val_split = 0.1, X_val = None, Y_val = None, shuffle = True, 
        feature_name = "ALL_FEATURES", recalculate_pickle = True, full = True, verbose = False):

        __MODEL_NAME = "Keras - ResNet50"
        __MODEL_FNAME_PREFIX = "KERAS_RESNET50/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])
        __history_params_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_params.csv"])
        __history_performance_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_history.csv"])
        __history_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_plot.png"])
        __model_architecture_plot_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, "_model_plot.png"])
        ##__scaler_file = "".join([nested_dir, feature_name, "_scaler.pkl"])

        if verbose: print("Retrieving model: %s..." % "".join([__MODEL_NAME, __MODEL_SUFFIX]))

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print("Pickle file for '%s' MODEL not found or skipped by caller." % feature_name)

            #act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
            act = 'adam'
            #lss = losses.mean_squared_error
            lss = 'mean_squared_error'
            #mtrc = [metrics.RootMeanSquaredError()]
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                mode = 'min', monitor = 'val_mae')

            # a few custom parameters for tuning
            include_top = True
            pooling = 'avg' # can be 'max' as well, only used if include_top is False

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):

                # Input image shape (H, W, C)
                input_img = Input(shape = (96, 96, 1))
                bn_axis = 3

                # Begin ResNet50
                x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_img)
                x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1') (x)
                x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
                x = Activation('relu')(x)
                x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
                x = MaxPooling2D((3, 3), strides=(2, 2))(x)

                x = self.__conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
                x = self.__identity_block(x, 3, [64, 64, 256], stage=2, block='b')
                x = self.__identity_block(x, 3, [64, 64, 256], stage=2, block='c')

                x = self.__conv_block(x, 3, [128, 128, 512], stage=3, block='a')
                x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='b')
                x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='c')
                x = self.__identity_block(x, 3, [128, 128, 512], stage=3, block='d')

                x = self.__conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
                x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
                x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
                x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
                x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
                x = self.__identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

                x = self.__conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
                x = self.__identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
                x = self.__identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

                if include_top:
                    x = GlobalAveragePooling2D(name='avg_pool')(x)
                    if full:
                        x = Dense(30, name='fc1000') (x)
                    else:
                        x = Dense(8, name='fc1000') (x)
                else:
                    if pooling == 'avg':
                        x = GlobalAveragePooling2D()(x)
                    elif pooling == 'max':
                        x = GlobalMaxPooling2D()(x)
                    else:
                        raise RuntimeError('The output shape of `ResNet50(include_top=False)` has been changed since Keras 2.2.0.')
                
                model = Model(input_img, x, name='resnet50')

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)


            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp], verbose = verbose)

            # print and/or save a performance plot
            self.__plot_keras_history(history = history, metric = 'mse', #metric = 'root_mean_squared_error', 
                model_name = __MODEL_NAME, feature_name = feature_name, file_name = __history_plot_file,
                verbose = verbose)
            
            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)
            if verbose: print("Model JSON, history, and parameters file saved.")

            # save a plot of the model architecture
            plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB', 
                show_shapes = True, show_layer_names = True, expand_nested = True, dpi=300)


        else:
            if verbose: print("Loading history and params files for '%s' MODEL..." % feature_name)
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)
            if verbose: print("Loading pickle file for '%s' MODEL from file '%s'" % (feature_name, __model_file_name))
            parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # inferencing
    def predict_keras_resnet50(self, X, feature_name = "unknown", full = True, verbose = False):

        __MODEL_NAME = "Keras - ResNet50"
        __MODEL_FNAME_PREFIX = "KERAS_RESNET50/"
        if full:
            __MODEL_SUFFIX = "_30"
        else:
            __MODEL_SUFFIX = "_8"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError("Model path '%s' does not exist; exiting inferencing." % nested_dir)

        __model_file_name = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".h5"])
        __model_json_file = "".join([nested_dir, feature_name, __MODEL_SUFFIX, ".json"])

        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n" % 
                (__model_file_name, __model_json_file))
        
        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for '%s'..." % (len(X), feature_name))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y
