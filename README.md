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

A few settings were altered briefly to examine how certain aspects of our code affected our output. These tests were done after an attempt was made to  resplit data to balance the loss and accuracy of our model. The re-split charts can be found below:

```Loss (re-split)```

![Loss chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/10%20Filters%20(Default)/24%20Epochs/2%20Pools%20(Standard)/Loss.png)

```Accuracy (re-split)```

![Accuracy chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/10%20Filters%20(Default)/24%20Epochs/2%20Pools%20(Standard)/Accuracy.png)

#### Changing Activation Functions
In one test, the final activation function was changed from softmax to sigmoid. Not much changed occurred from the default model.

```Loss (sigmoid output)```

![Loss chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-ReLU%3B%20ReLU-ReLU%20to%20Sigmoid/Loss.png)

```Accuracy (sigmoid output)```

![Accuracy chart for a seed of 4660 with sigmoid output](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-ReLU%3B%20ReLU-ReLU%20to%20Sigmoid/Accuracy.png)

In another test, we changed an couple of instances of ReLU activation layers to sigmoid, while maintaining a softmax output. While this eventually lead to a loss of zero for both training and validation, it tanked accuracy across the board.

```Loss (sigmoid activation)```

![Loss chart for a seed of 4660 with sigmoid functions](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-Sigmoid%3B%20ReLU-Sigmoid%20to%20Softmax/Loss.png)

```Accuracy (sigmoid activation)```

![Accuracy chart for a seed of 4660 with sigmoid functions](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/ReLU-Sigmoid%3B%20ReLU-Sigmoid%20to%20Softmax/Accuracy.png)

#### Changing Kernel Sizes
The kernel amounts were also modified, with one test setting them to 1 and another test setting them to 5. A run with a kernel size of 1 was much faster to process, but there was little benefit. A run with a kernel size of 5 was much slower to process, but it proved to be detrimental.

```Loss (kernel size of 1)```

![Loss chart for a seed of 4660 with a kernel size of 1](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/Kernel%20Size%20of%201%20(Much%20Faster)/Loss.png)

```Accuracy (kernel size of 1)```

[Accuracy chart for a seed of 4660 with a kernel size of 1](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/Kernel%20Size%20of%201%20(Much%20Faster)/Accuracy.png)

```Loss (kernel size of 5)```

![Loss chart for a seed of 4660 with kernel size of 5](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/Kernel%20Size%20of%205%20(Slower)/Loss.png)

```Accuracy (kernel size of 5)```

[Accuracy chart for a seed of 4660 with kernel size of 5](https://github.com/Robert-Beane/theoryProject/blob/scrape-for-test-data/Data%20Figures/Resplit%20-%208%20Images%20per%20Training%20Subcategory/Seed%20of%204660/Standard/Kernel%20Size%20of%205%20(Slower)/Accuracy.png)


### Regularization Attempts

After we began work on duplicating our data, we attempted to regularize our model through the use of L1 and L2 norms. The default model can be found in the results section (pre-duplication). The L1 and L2 norms are often used to curb overfitting, but they did not provide a significant improvement for our model.

#### L1 Regularization

```Loss (L1 regularization)```

[Accuracy chart for a seed of 4660 with L1 norm](https://lh3.googleusercontent.com/fife/AAWUweWgAkhBjMve4GGBevmNP5GjiuZij8Zxb9j-el2vCzKSn4XdN2IT-dwekLLqwYLzEY2OfF5JRklOnsdiPl2QZhiBcyrQ-dl5TW97qlCrsdPE-2wHzzBh_YtMRzOXoCjNMiaGbGulvQImsBl1oY4pykL13VaF1Y6xxFriEOoUfFWTh-bO1DFY7XER3CtQ0RliUxxCrr-Ph8YtCuaM6FIiShXSFna057Vn9Zk2jgO8ZMK4-209Sv2XzmHlTWErZQAsOPdpSjfOJNjOl_ZzpwtLjMPbOaWBhlYkOK0AEnPFNqASDZmpW0Avjg9DqJadTGZ4b0_QjDFW35c5IyPQgMgsJuNmW1jKyaYsc4n5uxvPTpqiU5OoWRorpNEDkQfqVGb96OFE7R_BQRtbJ_x5MXoccUZiiClMB7gYQy80d_lslJrhF0Ajm4952jC2jP3sDaWJByu-1QCZ7W8_DqA9CDL6xgkQEE4teRw9di1E3Bie8qlnKpJbgiA0nB5XqHucLuJgu6fqrZUfP60yGcA0sN17VDL8UWiJoGz0VlTGBAHaAvhWej94PoGq775AAqSgLaE-VB4nwhQ1MdnBOwfloF8YRmpAk-ae1uQ0PvhflS10vKZokesfMuUsMLwbC4WTyirP4Y-zhyJRhslaSAasbitVxNkj9NYU9d_Q4_Ay4UvTu8YUiSRm67Cjh4yd8XClMaqpmA9tyc1mAX5QB-RAkqtHt36hJtY74JVMF2ys1y5G5xEcm-J9bMWdnWrybiro_4hSMqkes1jm3qpfkL0jXVc=w1920-h927)

```Accuracy (L1 regularization)```

[Accuracy chart for a seed of 4660 with L1 norm](https://lh3.googleusercontent.com/fife/AAWUweVVBQmqiQdcqaXwgLwd__wlm3m1iqhKZBecMRTPMFkKGoiaOufcpqIrViKQkAdrZnD4Mv68wuHB2n5tiq2t1TYzRKDcn_6hjLOpXZORNVVKgPjRes6sA_aZh8_iSU7nkfeuUnk36pJ7ofk6fcZaiTEDsUh06RtrbSKoVQvOde08GdWkQl5oU6X2ZVbEP3Ib5Oh_i6RQHCJ7mUVGNKYg4riI16_VzaHl_TccfsHPYAg_i1Hp54iYBmlnOuyw81Jzfy5Ihvmcv-MbEHIzNxyZtMH2DAHhPVm9RWfjVdDnksrhFdxbv-R9BORwZ3kvO7O3KvcdccrAXRrcfqh7NAV5HHvfLBQoq0_rl8Hs4nUEkhwEV3bzy_dzvrilFoBMk6Z8wVtfZuqpYuJaVM-1CNTIBRvFl_J0e3WRSTOWyZKQG5cM3auVIL7Ws9jswdMyn9WNNJUNJfEgoo1Pqd9YmsXNNp2WN7iNoBRne4kzB9A7K-jEK-jIvgOLT8kjCk5euWy1XvAjYJ5v3JzJrro54AQE8fUTAB_cNwPzpFxebVskqfVPyh1Bg5xLZCK8Db3YT0eMx2CzDiqpdS_F1NLobG7WheeOVINXWiz3ov1puqnCGndLY5_J5wH5SmYgDkL1DJgMJuHn5wl4TG-BP7uB_iALBpNDgALnECZYlvZj9NvP50Veg0dXAfIXO6MvU_HRfapjZrW-gDiYB77bNZpovJodWDIghJPYhFrNMM86AeAIb4Voy-vyTc-upSm2FuSATKsJT5EcX8jdgD1hTPGKw5w=w1920-h927)

#### L2 Regularization

```Loss (L2 regularization)```

[Accuracy chart for a seed of 4660 with L2 norm](https://lh3.googleusercontent.com/fife/AAWUweU3MtS3o6VQHniTTEww9msuXAsT0Z_1xablA6o2XE_HNjmWMmoGe0GNwlYU_4jSLkiN6ot0qJP_5NKv2UeUUR3pR38GOyYd68cHkYTfsaXYDcI8932qn4yvBwc0dcTBU4P7EWp_UrkRMu5RphJg-Y-CLXYWqm4iJzh4BZRdJg5miCctsCd7Rk18HqWWKyeCnbbUdmwMlqTYVEhOJTxWa_os1dI7A0A59U_lRjRCcUBaEo8YwvCegqnHMKDSVLc73Mqc8Bca1M6D-RyxZQ5_CEwpdqyloEt8_R2UDNsEN2bTr9eKJq0Hg4YJ5YnTiszl95BDR7K9ZFavLIOqxLfpuNgtq1qHqY7p0ffekm4a8PHG5zitWLWLyaAEJrHY8YeSlQraO6XWljXBd7pTQz7_vj84tcslHgWfxtu1vZOFAVrWlLBRBYiaWN54aqsGRo6thqKwQb0hGpPKJyQnQqaiPEq6g5RgKJF6l3-yavGHRfcBuhAXLELlxpWX5parXK91dbJaGTGIKSZzdMrr3P8kAoVMmy2wTBic9LM-2fxh-B-xM9VTBjdEBSe4H5DPnj5j1E3sPbbqYID0wNCs9jzNfie2El08rDRQC_BGG4iKJEsSOEvaudvgVL7M-qWh9CXOX1UL8FN9sdKlqNiz38FXK9BPCCaGpPYifsYtktAdguY869TLbOl8bhKL3bFk-8ueRFj6F7xJYlwKnBG92K1RahYcq4gUp58LF3bItobvoH_5uBabzFITLKvdle6S5JGF3DpWyaanhyUPpSwh6Pc=w1920-h526)

```Accuracy (L2 regularization)```

[Accuracy chart for a seed of 4660 with L2 norm](https://lh3.googleusercontent.com/fife/AAWUweW1hoz3ftPxon2A3FAZ0zhzN6QkCZy94ODIpqIqqJ-teMgxjrRLGTa8Xw_OkMjfVX7mtD7bQbqAk1y_BcrNWCX3FnYmMoWKP379ks3KOgsQb10HHHZpbFqiVzytRE6FfngD9GDaIGW1iAeVVWQHGHfkiqlGu0KTSdUhXYwC2sxkyLB1_agv1mfEhgcS8qiN5hrs51fQ0h_JJOC8JhT5YZvkU24bGkXuTv7I1lubgWOigfyPjD8bszYyIKfUV3LKrFjZoUzAx7rSEeQSaN-fd4Dc4Z4IXQh_5lzIGQqRrq8INw7NU3ykF8kyHWa5AZc0zJ0D92Hy_4WqU5MA3pjsX7zQiK_0htI7q64qXm6GMqBV7fHsnorJrc-AyQwAnRGkEhQbbftV0X_fJ1AmBnNkmZriZRVjODPe1aGlKb-MFkmMw5yU_D1Xx2PLbGCnveH4fh3dGziJ8-ierirtIOTO15-pJy-MeM5emuVUwU6IFDqISRN711TDnIEE9BdW3RL2R_P_cM__AXpXinMUDpVVZRxVxZdsgDdaaqIwMlozqRJ8k2M-sjcMb5BlIf9gm6DSxxprTex0TAgx5KFIVumm5TUJxYlmMPk-EmzEpgVr4eAqjqHvyqGpNp2hEGqIyiyd18AyIpziQkwSJTB91o8zHqX_yC-GB6JOdjL7tENLI_7H4s48uO_Hr90q-fM48asT_EPs18cdO7Sl5vrGIoBSTu2FDv-ThvsfDff0d93DCNNbqFwwnWBovgfvL9EwOS_OJXj8zYmICdQxABs0Iog=w1920-h526)

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
