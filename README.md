# Cancer Classifier using CNN

## Decscription

This is an exploration into CNNs for classification of data and working with biological data specficially cancer images

## Goal

The goal of this project is to create a model that is able to classify cancer cells with 80% accuracy based on the given data set

## Data

This data set uses images found on kaggle at
[[https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download]]
Which is classified into 5 categories: - Lung benign tissue - Lung adenocarcinoma - Lung squamous cell carcinoma - Colon adenocarcinoma - Colon benign tissue

## Status

### Progress

The data pipeline has been set up along with the model and the accuracy tests to see how well the model is preforming

## Issues

Currently the accuracy sits around 30% and has large miss classifications across both lung and colon images. 

## Future Work

Will try to see if the model can start by identifying the difference between benign lung tissue and benign colon tissue to see if it is just a sampling issue. Will likely have to change to a binary classification for this step. 

After trying tweaking the model itself I think going forward instead of using a self-trained model using only 25000 images for testing instead starting with an already trained model and then training it on the data set to see if the results improve. Due to the lack of data and quality data it could be affecting the results and not allowing for the model to generalize properly.
