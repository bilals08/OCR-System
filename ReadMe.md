# TRS: Object Character Recognition using HOG Feature Extractor and Random Forest Classifier

This is an implementation of OCR (Object Character Recognition) using the HOG (Histogram of Oriented Gradients) feature extractor and Random Forest Classifier. The dataset used for training on English Alphabet Dataset.

## Overview
The HOG feature extractor is used to extract features from the images of the English Alphabet Dataset. The HOG descriptor is a popular feature descriptor used in computer vision and image processing for object detection and recognition. It captures the local structure of the image by computing the distribution of gradient orientations in small subregions of the image.

The Random Forest Classifier is then used to train a model on the extracted features. Random Forest is an ensemble learning method that operates by constructing multiple decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## Requirements
* Python 3.x
* OpenCV
* Scikit-learn

## Usage
Run ClassifierOCR.py after setting the path

## Contributing
Contributions to this project are welcome. Please feel free to create an issue or a pull request if you have any suggestions or improvements.

