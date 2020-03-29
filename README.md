Facial Keypoints Detection
===========================
**Authors** : [Cristopher Benge](https://cbenge509.github.io/) | [William Casey King]([https://jackson.yale.edu/person/casey-king/](https://jackson.yale.edu/person/casey-king/)) 
U.C. Berkeley, Masters in Information & Data Science program - [datascience@berkeley]([https://datascience.berkeley.edu/](https://datascience.berkeley.edu/)) 
Spring 2020, W207 - Machine Learning - D. Schioberg, PhD
Section 5 - Wed. 4:00pm PDT

## Description

This repo contains a solution for the [Kaggle Facial Keypoints Detection]([https://www.kaggle.com/c/facial-keypoints-detection](https://www.kaggle.com/c/facial-keypoints-detection)) challenge, achieving a best score of **~1.42 RMSE**, which is good for a 4th place position on the [now locked] private leaderboard:

![img](/images/top10.jpg)

## Evaluation Criteria

The Kaggle facial keypoints detection challenge asks competitors to identify, with as small an error as possible, the (x,y) point coordinates for up to 15 facial landmarks (left center eye, nose tip, left outer eyebrow, etc) on 1,783 images in the provided TEST dataset.  All images are sized as 96x96x1 grayscale.  Each point is submitted as their individual X and Y coordinate values for separate scoring, thus competitors will need to submit up to 30 values for an image containing all 15 landmarks.

The competition is ranked by the lowest error in prediction by the [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation):
$$
\sqrt{\left(\frac{1}{n}\right)\sum_{i=1}^{n}(y_{i} - x_{i})^{2}}
$$

## Training Pipeline
```mermaid
graph TD;
	Train(Raw Train Data)
	style Train fill:#fdb515,stroke:#000333,stroke-width:2px 
	Train30[Complete Set]
	style Train30 fill:#fdb515,stroke:#000333,stroke-width:2px 
	Train8[Partial Set]
	style Train8 fill:#fdb515,stroke:#000333,stroke-width:2px 
	Augmentation[Cleaning & Augmentation]
	style Augmentation fill:#fdb515,stroke:#000333,stroke-width:2px
	ConvNet5[Conv2D 5-Layer]
	style ConvNet5 fill:#00b0da,stroke:#000333,stroke-width:2px
	NaimishNet[NaimishNet]
	style NaimishNet fill:#00b0da,stroke:#000333,stroke-width:2px
	Kaggle1[Conv2D 10-Layer]
	style Kaggle1 fill:#00b0da,stroke:#000333,stroke-width:2px
	Kaggle2[Local2D]
	style Kaggle2 fill:#00b0da,stroke:#000333,stroke-width:2px
	InceptionV1[Inception V1]
	style InceptionV1 fill:#00b0da,stroke:#000333,stroke-width:2px
	InceptionV3[Inception V3]
	style InceptionV3 fill:#00b0da,stroke:#000333,stroke-width:2px
	LeNet5[LeNet 5-Layer]
	style LeNet5 fill:#00b0da,stroke:#000333,stroke-width:2px
	CV_Stacker(Generalized Stacker - KFold)
	style CV_Stacker fill:#9DAD33,stroke:#f66,stroke-width:3px,color:#ffffff,stroke-dasharray: 5, 5
	Predict8[Predict Partial]
	style Predict8 fill:#C2B9A7,stroke:#000333,stroke-width:2px
	PredictNot8[Predict Complete]
	style PredictNot8 fill:#C2B9A7,stroke:#000333,stroke-width:2px
	Merged[Merged Predictions]
	style Merged fill:#B9D3B6,stroke:#000333,stroke-width:2px
	FeatureInteractions[Feature Interactions]
	style FeatureInteractions fill:#B9D3B6,stroke:#000333,stroke-width:2px
	Metaregressor>Level 2 Metaregressor]
	style Metaregressor fill:#f9f,stroke:#333,stroke-width:4px 
	
	subgraph Data Preparation
	Train-- all 30 labels present -->Train30;
	Train-- only 8 labels present -->Train8;
	Train30 --> Augmentation;
	Train8 --> Augmentation;
	end

	subgraph Generalized Stacking
	Augmentation-.-> ConvNet5;
	Augmentation-.-> NaimishNet;
	Augmentation-.-> Kaggle1;
	Augmentation-.-> Kaggle2;
	Augmentation-.-> InceptionV1;
	Augmentation-.-> InceptionV3;
	Augmentation-.-> LeNet5;
	Augmentation-.-> ConvNet5;
	Augmentation-.-> NaimishNet;
	Augmentation-.-> Kaggle1;
	Augmentation-.-> Kaggle2;
	Augmentation-.-> InceptionV1;
	Augmentation -.-> InceptionV3;
	Augmentation-.-> LeNet5;
	ConvNet5-- K=5 ---CV_Stacker;
	NaimishNet-- K=5 ---CV_Stacker;
	Kaggle1-- K=5 ---CV_Stacker;
	Kaggle2-- K=5 ---CV_Stacker;
	InceptionV1-- K=5 ---CV_Stacker;
	InceptionV3-- K=5 ---CV_Stacker;
	LeNet5-- K=5 ---CV_Stacker;
	CV_Stacker --> Predict8;
	CV_Stacker --> PredictNot8;
	Predict8-.-Merged;
	PredictNot8-.-Merged;
	Merged -.-> FeatureInteractions;
	end

	FeatureInteractions --> Metaregressor;	
```

## Inference Pipeline
```mermaid
graph TD;
	Test(Raw Test Data)
	style Test fill:#fdb515,stroke:#000333,stroke-width:2px 
	TestOther[Not 8 Labels]
	style TestOther fill:#fdb515,stroke:#000333,stroke-width:2px 
	Test8[8 Labels Only]
	style Test8 fill:#fdb515,stroke:#000333,stroke-width:2px 
	ConvNet5[Conv2D 5-Layer]
	style ConvNet5 fill:#00b0da,stroke:#000333,stroke-width:2px
	NaimishNet[NaimishNet]
	style NaimishNet fill:#00b0da,stroke:#000333,stroke-width:2px
	Kaggle1[Conv2D 10-Layer]
	style Kaggle1 fill:#00b0da,stroke:#000333,stroke-width:2px
	Kaggle2[Local2D]
	style Kaggle2 fill:#00b0da,stroke:#000333,stroke-width:2px
	InceptionV1[Inception V1]
	style InceptionV1 fill:#00b0da,stroke:#000333,stroke-width:2px
	InceptionV3[Inception V3]
	style InceptionV3 fill:#00b0da,stroke:#000333,stroke-width:2px
	LeNet5[LeNet 5-Layer]
	style LeNet5 fill:#00b0da,stroke:#000333,stroke-width:2px

	ConvNet5_2[Combined Output]
	style ConvNet5_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	NaimishNet_2[Combined Output]
	style NaimishNet_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	Kaggle1_2[Combined Output]
	style Kaggle1_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	Kaggle2_2[Combined Output]
	style Kaggle2_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	InceptionV1_2[Combined Output]
	style InceptionV1_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	InceptionV3_2[Combined Output]
	style InceptionV3_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	LeNet5_2[Combined Output]
	style LeNet5_2 fill:#B9D3B6,stroke:#000333,stroke-width:2px,stroke-dasharray: 5, 5
	Merged[Merged Predictions]
	style Merged fill:#9DAD33,stroke:#000333,stroke-width:2px
	FeatureInteractions[Feature Interactions]
	style FeatureInteractions fill:#9DAD33,stroke:#000333,stroke-width:2px
	Metaregressor(Level 2 Metaregressor)
	style Metaregressor fill:#00b0da,stroke:#333,stroke-width:2px 
	Predictions>Final Predictions]
	style Predictions fill:#f9f,stroke:#333,stroke-width:4px 
	
	subgraph Preprocessing
	Test-- 2-6, 10-30 labels required -->TestOther;
	Test-- only 8 labels required -->Test8;
	end

	subgraph Level 1 Predictions
	TestOther-.-> ConvNet5;
	TestOther-.-> NaimishNet;
	TestOther-.-> Kaggle1;
	TestOther-.-> Kaggle2;
	TestOther-.-> InceptionV1;
	TestOther-.-> InceptionV3;
	TestOther-.-> LeNet5;
	Test8-.-> ConvNet5;
	Test8-.-> NaimishNet;
	Test8-.-> Kaggle1;
	Test8-.-> Kaggle2;
	Test8-.-> InceptionV1;
	Test8-.-> InceptionV3;
	Test8-.-> LeNet5;
	end

	subgraph Level 2 Predictions
	ConvNet5-- 8-label--> ConvNet5_2;
	ConvNet5-- Not 8-label --> ConvNet5_2;
	NaimishNet--8-label--> NaimishNet_2;
	NaimishNet--Not 8-label--> NaimishNet_2;
	Kaggle1--8-label--> Kaggle1_2;
	Kaggle1--Not 8-label--> Kaggle1_2;
	Kaggle2--8-label--> Kaggle2_2;
	Kaggle2--Not 8-label--> Kaggle2_2;
	InceptionV1--8-label--> InceptionV1_2;
	InceptionV1--Not 8-label--> InceptionV1_2;
	InceptionV3--8-label--> InceptionV3_2;
	InceptionV3--Not 8-label--> InceptionV3_2;
	LeNet5--8-label--> LeNet5_2;
	LeNet5--Not 8-label--> LeNet5_2;

	ConvNet5_2 -.- Merged;
	NaimishNet_2 -.- Merged;
	Kaggle1_2 -.- Merged;
	Kaggle2_2 -.- Merged;
	InceptionV1_2 -.- Merged;
	InceptionV3_2 -.- Merged;
	LeNet5_2 -.- Merged;
	Merged -.-> FeatureInteractions;
	FeatureInteractions --> Metaregressor;
	end

	Metaregressor --> Predictions;
```