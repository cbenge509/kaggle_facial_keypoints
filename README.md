Facial Keypoints Detection
===========================

![GitHub](https://img.shields.io/github/license/cbenge509/kaggle_facial_keypoints) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/kaggle_facial_keypoints/tensorflow) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/kaggle_facial_keypoints/keras)

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

## Inference Pipeline
![inference pipeline](/images/inference_pipeline.jpg)
