# Detecting LEGO minifigures with TensorFlow

Dataset Source : [Kaggle LEGO minifigures](https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification)

## First Run Instructions
Before you can run the Convolutional Neural Network (CNN), you have to install the required dependencies.
To do so, run the following command in your terminal.

```pip install -r requirements.txt```

You also must install Graphviz which can be done [here](https://graphviz.org/download/).

### Files
* [normalizeTraining.py](normalizeTraining.py)  - Normalizes all of the images contained din the training folders.
* [multiLayer.py](multiLayer.py)  - Runs implementation of Multi Layer Perceptron (low accuracy).
* [convolutional.py](convolutional.py)  - Runs the CNN for the LEGO dataset; the main implementation.

## Overview

### Design Matrix
Our design matrix consists of three categories of Lego minifigures: Star Wars, Marvel, and other. From there, 37 subcategories are distributed among them, numbered starting at 0001 and generally consisting of less than thirty 512 by 512 images a piece (after duplication). These subcategories identify characters that Lego has made minifigures for (e.g., Yoda, Captain America, Harry Potter, etc.). To identify them more broadly, each category is labeled with a number from 1-37 in the metadata csv file.

The images are preprocessed before training through normalization and downsizing by a scale of two (i.e., 512 x 512 down to 256 x 256). It is possible to run the neural network without any preprocessing, but this significantly increases the training and validation times, as well as the quality of results.

### Goals and Hypothesis

Our primary goal with this project was to be able to accurately process the training and validation data above a threshold of 20%. Our hypothesis was that we would eventually be able to balance out these accuracies through tweaks to our neural network design (different filter counts, activation functions, kernel sizes, etc.).

### Project Methods

### Training Process

## Design Process

## Results

## Challenges

## Opportunities for Improvement
