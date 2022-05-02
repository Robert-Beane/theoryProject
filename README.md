# Detecting LEGO minifigures with TensorFlow
## Robert Beane, Kyle Day, Jacob Jenness

# Table of Contents
* [LEGO Dataset](#Dataset)
* [First Run Instructions](#First-Run-Instructions)
* [File Overview](#Files)
* [Overview](#Overview)
    * [Design Matrix](#Design-Matrix)
    * [Sample Rows](#Sample-Rows)
    & [Model Summary](#Model-Summary)
    * [Goals and Hypothesis](#Goals-and-Hypothesis)
    * [Project Methods](#Project-Methods)
    * [Training Process](#Training-Process)
* [Design Process](#Design-Process)
* [Results](#Results)
* [Challenges](#Challenges)
* [Opportunities for Improvement](#Opportunites-for-Improvement)

## Dataset
Dataset Source : [Kaggle LEGO minifigures](https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification)

## First Run Instructions
Before you can run the Convolutional Neural Network (CNN), you have to install the required dependencies.
To do so, run the following command in your terminal.

```pip install -r requirements.txt```

You also must install Graphviz which can be done [here](https://graphviz.org/download/).

## Files
* [normalizeTraining.py](normalizeTraining.py)  - Normalizes all of the images contained in the training folders.
* [multiLayer.py](multiLayer.py)  - Runs implementation of Multi Layer Perceptron (low accuracy).
* [convolutional.py](convolutional.py)  - Runs the CNN for the LEGO dataset; the final, main implementation.

## Overview

### Design Matrix
Our design matrix consists of three categories of Lego minifigures: Star Wars, Marvel, and other. From there, 37 subcategories are distributed among them, numbered starting at 0001 and generally consisting of less than thirty 512 by 512 images a piece (after duplication). These subcategories identify characters that Lego has made minifigures for (e.g., Yoda, Captain America, Harry Potter, etc.). To identify them more broadly, each category is labeled with a number from 1-37 in the metadata csv file.

The images are preprocessed before training through normalization and downsizing by a scale of two (i.e., 512 x 512 down to 256 x 256). It is possible to run the neural network without any preprocessing, but this significantly increases the training and validation times, as well as the quality of results.

#### Sample Rows
#### ```index.csv```
|path               |class_id|
|-------------------|--------|
|marvel/0001/001.jpg|1       |
|marvel/0001/002.jpg|1       |
|marvel/0001/003.jpg|1       |

#### ```test.csv```
|path               |class_id|
|-------------------|--------|
|test/001.jpg       |32      |
|test/002.jpg       |32      |
|test/003.jpg       |32      |

#### ```metadata.csv```
|class_id           |lego_ids|lego_names               |minifigure_name|
|-------------------|--------|-------------------------|---------------|
|1                  |[76115] |['Spider Mech vs. Venom']|SPIDER-MAN     |
|2                  |[76115] |['Spider Mech vs. Venom']|VENOM          |
|3                  |[76115] |['Spider Mech vs. Venom']|AUNT MAY       |

### Model Summary
```
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input (InputLayer)          [(None, 256, 256, 3)]     0         
                                                                 
 conv_1_1 (Conv2D)           (None, 254, 254, 10)      280       
                                                                 
 conv_1_2 (Conv2D)           (None, 252, 252, 10)      910       
                                                                 
 max_pool_1 (MaxPooling2D)   (None, 126, 126, 10)      0         
                                                                 
 conv_2_1 (Conv2D)           (None, 124, 124, 10)      910       
                                                                 
 conv_2_2 (Conv2D)           (None, 122, 122, 10)      910       
                                                                 
 max_pool_2 (MaxPooling2D)   (None, 61, 61, 10)        0         
                                                                 
 flatten_layer (Flatten)     (None, 37210)             0         
                                                                 
 output_layer (Dense)        (None, 36)                1339596   
                                                                 
=================================================================
```

### Goals and Hypothesis

Our primary goal with this project was to be able to accurately process the training and validation data above a threshold of 20%. Our hypothesis was that we would eventually be able to balance out these accuracies through tweaks to our neural network design (different filter counts, activation functions, kernel sizes, etc.).

### Project Methods

Our final implementation utilizes a CNN (convolutional neural network) to train and validate data. This neural network is particulary effective for image processing. Images consist of multidimenionsal data which can be quite complex to process and analyze. CNNs are able to reduce the dimensionality of images by taking portions of the data and filtering it through their networks of layers. A great amount of precision can be acquired by implementing a CNN structure. Once we did this for our own project, our training data went from being ~5% accurate to ~%90 accurate.

### Training Process

## Design Process

## Results

## Challenges

## Opportunities for Improvement
