Facial Keypoints Detection
===========================

![GitHub](https://img.shields.io/github/license/cbenge509/kaggle_facial_keypoints) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/kaggle_facial_keypoints/tensorflow) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/kaggle_facial_keypoints/keras) ![GitHub contributors](https://img.shields.io/github/contributors/cbenge509/kaggle_facial_keypoints) ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/cbenge509/kaggle_facial_keypoints?include_prereleases)

<img align="right" width="180" src="./images/ucb.png"/>

#### Authors : [Cristopher Benge](https://cbenge509.github.io/) | [William Casey King](https://jackson.yale.edu/person/casey-king/) 

U.C. Berkeley, Masters in Information & Data Science program - [datascience@berkeley](https://datascience.berkeley.edu/) 

Spring 2020, W207 - Machine Learning - D. Schioberg, PhD <br>
Section 5 - Wed. 4:00pm PDT

## Description

This repo contains a solution for the [Kaggle Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection) challenge, developed by Cristopher Benge (U.C. Berkeley) and William Casey King (Yale University, U.C. Berkeley).  This solution leverages a variety of neural network architectures and a custom K-fold generalized stacker to achieve a top 5 score.

#### Highlight of key files included in this repository:

  |File | Description |
  |:----|:------------|
  |[Facial Keypoints - EDA.ipynb](Facial%20Keypoints%20-%20EDA.ipynb) | Jupyter Notebook containing exploratory data analysis findings of the competition data.|
  |[prepare_data.py](prepare_data.py) | Performs preprocessing of TRAIN and TEST datasets.|
  |[train_model.py](train_model.py) | Trains one or more models based on user-provided arguments.|
  |[predict.py](predict.py) | Predict from last trained version of one or more user-specified models.|
  |[train_stack.py](train_stack.py) | Performs K-Fold generalized stacking using the user-specified models.|
  |[predict_stack.py](train_stack.py) | Performs metaregressor traning and prediction from all stacked model files.|
  |[generate_submission.py](prepare_data.py) | Generates submissions of individual models by combining the 8-only and Not-8-ony predictions.|
  |[data_loader.py](/utils/data_loader.py) | Utility class used for loading the competition dataframes.|
  |[data_transformer.py](/utils/data_transformer.py) | Utility class for performing data cleaning and augmentation.|
  |[model_zoo.py](utils/model_zoo.py) | Utility class; provides model construction from the model zoo construction and base prediction routines.|
  

## Performance

This solution achieves a best score of **~1.42 RMSE**, which is good for a 4th place position on the [now locked] private leaderboard:

<img src="/images/top10.jpg" width="400"/>

## Evaluation Criteria

The Kaggle facial keypoints detection challenge asks competitors to identify, with as small an error as possible, the (x,y) point coordinates for up to 15 facial landmarks (left center eye, nose tip, left outer eyebrow, etc) on 1,783 images in the provided TEST dataset.  All images are sized as 96x96x1 grayscale.  Each point is submitted as their individual X and Y coordinate values for separate scoring, thus competitors will need to submit up to 30 values for an image containing all 15 landmarks.

The competition is ranked by the lowest error in prediction by the [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation):

<img src="/images/rmse.jpg" width="400" />

## Training Pipeline

![train pipeline](/images/train_pipeline.jpg)

Our training pipeline begins with splitting the TRAIN and TEST datasets into two separate challenges (more on the reason for this below), applying simple cleaning, and adding a moderate amount of augmentation images derived from the initial, base images.  We experiemented with a lot of augmentation strategies, but in the end the following performed best for our models:

* manual label assignment to ~56 training labels that were overtly incorrect.
* Image pixel values normalized between 0 and 1 (both TRAIN and TEST).
* positive and negative rotation of images by the following degrees: {+/-5, +/-9, +/-13}
* elastic transformation of images using alpha/sigma values of {991/8, 1201/11}.
* random gaussian noise, applying between ~-0.03 to ~+0.03 values to the pixels.
* brightened images by the following scalars: {1.2x, 1.5x}.
* dimmed images by the following scalers: {0.6x, 0.8x}.
* contrast stretching in the following ranges: {0.0 - 1.5, 0.0 - 2.0}.
* custom image sharpening 3x3 conv2d kernel applied one time.

Many other augmentations were experimented with, such as horizontal flipping, blurring, laplacian smoothing, etc. but all resulted in adverse impact to predict error.

The TRAIN dataset provided by the competition organizers appears to derive from two distinct sources: one where all 15 landmark labels are present, and another where only 4 are present (note: there are a few images thorughout that have some other count of missing label information, but they are the exception).  Initially, we were able to achieve a single best-model RMSE score of **1.62800** by addressing all missing labels through interpoloation of their `np.nanmean()` keypoint average location.

We discovered significant improvement (approx. _-0.2 RMSE_) through splitting our training pipeline in two: (1) trained models on the set with all data points available, and (2) trained models for only the 4 keypoints present in the "partial" set.  At prediction time, we use the "partial" models to predict only those TEST records for which we are asked to predict 4 points, and use the "complete" (1) models for all other facial keypoints.  Controlling for all other changes, this move alone resulted in a best single-model score of RMSE **1.43812**.

Generalized stacking using a K-Fold approach was used to avoid overfitting at the metaregressor phase.  Seven neural networks were used as Level 1 predictors for both the "complete" and "partial" data sets (the models were identical save for the final layer).  All models were trained using a batch size of 128 * 2 (128 for each GPU detected) and epoch size of 300.  Training occurred on machines with either 2 x NVidia RTX 2080 Ti's or 2 x NVidia Tesla V100's.  A general descripton for each Level 1 model is listed below:

| Model Name | Description |
|:-----------|:------------|
|Conv2D 5-Layer | A simple 5-layer conv2d network  |
|NaimishNet | A 7.4M parameter convnet that  learns only one keypoint at a time.  |
|Conv2D 10-Layer | A deeper version of Conv2D 5-Layer |
|Local2D | A modified version of Conv2D whose final layer is a Local2D and global average pool |
|Inception V1 | A modified version of Google's inception V1 |
|Inception V3 | A modified version of Google's inception V3 |
|LeNet5 | A slightly modified 5-layer version of the class LeNet |

Following K-Fold training of the seven models for both the "complete" TRAIN dataset and the "partial" TRAIN dataset, all sevel models are then trained again on the complete TRAIN dataset for the prediction phase, used during final inferencing.  Aft3er all L1 training predictions are captured, simple multiplciation feature interactions are generated per model, resuliting in a per-model input space increase from 30 (the initial 30 keypoint X and Y coordinates) to 30 + (30-choose-2) feature interactions = 465 per model inputs.  A MultiTaskElasticNet linear regression biased to L1 regularization @ 100% is used as our final regressor over all (465 * 7) = 3,255 inputs.  This model is then saved and used in our final inferencing for submission.

## Inference Pipeline
![inference pipeline](/images/inference_pipeline.jpg)

The inferencing pipeline behavior is essentially duplicative of the training process above, with the exception that TEST is also split based on whether we are scored on finding 4 keypoints (8 labels) or NOT 4 keypoints (Not 8 labels).  These images are inferenced through all sevel Level 1 models, combined, feature interactions are calculated, and a final submission inference is taken from the Level 2 MultiTaskElasticNet linear regressor.

License
-------
Licensed under the MIT License. See [LICENSE](LICENSE>txt) file for more details.