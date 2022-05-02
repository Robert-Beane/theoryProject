# Detecting LEGO Minifigures with TensorFlow
## Robert Beane, Kyle Day, Jacob Jenness

# Table of Contents
* [LEGO Dataset](#Dataset)
* [First Run Instructions](#First-Run-Instructions)
* [File Overview](#Files)
* [Overview](#Overview)
    * [Design Matrix](#Design-Matrix)
    * [Sample Rows](#Sample-Rows)
    * [Model Summary](#Model-Summary)
    * [Goals and Hypothesis](#Goals-and-Hypothesis)
    * [Project Methods](#Project-Methods)
    * [Training Process](#Training-Process)
* [Design Process](#Design-Process)
* [Results](#Results)
* [Challenges](#Challenges)

## Dataset
Dataset Source : [Kaggle LEGO Minifigures](https://www.kaggle.com/datasets/ihelon/lego-minifigures-classification)

## First Run Instructions
Before you can run the Convolutional Neural Network (CNN), you have to install the required dependencies.
To do so, run the following command in your terminal.

```pip install -r requirements.txt```

You also must install Graphviz which can be done [here](https://graphviz.org/download/).

## Files
* [normalizeTraining.py](normalizeTraining.py)  - Script for normalizing images, was not used in the final CNN.
* [multiLayer.py](multiLayer.py)  - Runs implementation of Multi Layer Perceptron (low accuracy).
* [convolutional.py](convolutional.py)  - Runs the CNN for the LEGO dataset; the final, main implementation.

## Overview

### Design Matrix
Our design matrix consists of three categories of LEGO Minifigures: Star Wars, Marvel, and other. From there, 37 subcategories are distributed among them, numbered starting at 0001 and generally consisting of less than thirty 512 by 512 images a piece (after duplication). 
These subcategories identify characters that LEGO has made Minifigures for (e.g., Yoda, Captain America, Harry Potter, etc.). To identify them more specifically, each subcategory is labeled with a number from 1-37 in the three CSV files included.

The images are preprocessed before training through downsizing by a scale of two (i.e., 512 x 512 down to 256 x 256). It is possible to run the neural network without any preprocessing, but this significantly increases the training and validation times, as well as the quality of results.

### Sample Rows
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

Our final implementation utilizes a CNN (convolutional neural network) to train and validate data. This neural network is particulary effective for image processing. Images consist of multidimensional data which can be quite complex to process and analyze. 
CNNs are able to reduce the dimensionality of images by taking portions of the data and filtering it through their networks of layers. A great amount of precision can be acquired by implementing a CNN structure. Once we did this for our own project, our training data went from being ~5% accurate to ~90% accurate.

### Training Process

Our dataset already came with a set for training (```index.csv```) that contained the ```class_id``` and ```path``` for the image. The set already came with a pre-done testing dataset (```test.csv```)
This file had the same format as ```index.csv``` but consisted of randomly selected images of specific characters. We attempted adding our own images of the same characters to the training dataset but found that our accuracy became much more inconsistent. We also decided to use our testing dataset as our validation dataset. This could also be a cause for the low validation accuracy we see in our network.

## Design Process

### Code Tinkering

A few settings were altered briefly to examine how certain aspects of our code affected our output. For the default model, refer to the results section.

#### Changing Activation Functions
In one test, the final activation function was changed from softmax to sigmoid. This modification appeared to prolong the validation loss and delay the training loss. In addition, it appeared to increase our validation accuracy slightly.
```Loss (sigmoid output)```
![Loss chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-ReLU%3B%20ReLU-ReLU%20to%20Sigmoid/Loss.png)

```Accuracy (sigmoid output)```
![Accuracy chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-ReLU%3B%20ReLU-ReLU%20to%20Sigmoid/Accuracy.png)

In another test, we changed an couple of instances of ReLU activation layers to sigmoid, while maintaining a softmax output. While this eventually lead to a loss of zero for both training and validation, it tanked accuracy across the board.

![Loss chart for a seed of 4660 with sigmoid functions](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-Sigmoid%3B%20ReLU-Sigmoid%20to%20Softmax/Loss.png)

![Accuracy chart for a seed of 4660 with sigmoid functions](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-Sigmoid%3B%20ReLU-Sigmoid%20to%20Softmax/Accuracy.png)

#### Changing Filter Amounts

### Regularization Attempts

#### L1 Regularization

#### L2 Regularization

## Results

When running our network we discovered that our accuracy is pretty high and our loss is pretty low. However, our validation accuracy is relatively low compared to training and our validation loss is much higher than the training loss.
Because of this observation, we tried duplicating our testing data so that we have a higher percentage of training vs testing. For accuracy, we discovered that the accuracy rose quicker than before but our validation accuracy was still low. 
The same could be said for the training loss and validation loss. Our results for each case can be seen in the charts below.
#### ```Loss (not duplicated training data)```
![Loss chart for a seed of 4660 without duplicated training data](Charts/4660LossTrain.png)

#### ```Accuracy (not duplicated training data)```
![Accuracy chart for a seed of 4660 without duplicated training data](Charts/4660AccuracyTrain.png)

#### ```Loss (duplicated training data)```
![Loss chart for a seed of 4660 with duplicated training data](Charts/4660LossDuplicatedTrain.png)

#### ```Accuracy (duplicated training data)```
![Accuracy chart for a seed of 4660 with duplicated training data](Charts/4660AcurracyDuplicatedTrain.png)

We think this is a result of overfitting. We believe this is due to the number of categories we have and how we only have a few images per category.
If our dataset had more images of each minifig, we think that we could get our validation accuracy higher and our validation loss lower.

## Challenges
Our main challenge was working with as little data as we had. The best image recognition neural networks take in thousands images for training and validation each - but this data set only has a few hundred images. While we managed to significantly upgrade our training accuracy in the development of the CNN, 
there always seemed to be a low asymptote which our validation data could never go over once we reached a certain threshold. The most substantial way to improve this model would be to find more images of minifigures included to train on and validate.
