#  start of main training
import pandas as pd
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Update this if you didnt install Graphviz to Program Files
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import datetime


#  Setting up randomized seed
seed = 4660
random.seed(seed)
np.random.seed(seed)

TRAIN_DATA = pd.read_csv('index.csv')
TEST_DATA = pd.read_csv('test.csv')
METADATA = pd.read_csv('metadata.csv')

# prints to show data
print(TRAIN_DATA.shape)
print(TRAIN_DATA.sample(5))

print(TEST_DATA.shape)
print(TEST_DATA.sample(5))

print(METADATA.shape)
print(METADATA.sample(5))

# merge csv files
TRAIN_DATA = TRAIN_DATA.merge(METADATA, on='class_id')
TEST_DATA = TEST_DATA.merge(METADATA, on='class_id')
print(TRAIN_DATA.head())

PATH = '../theoryProject/'  # make sure this path is accurate to your own install
IMG_SIZE = 256


def tf_load_data(dataframe, batch_size=32, img_size=IMG_SIZE, directory_path=PATH, rescale=True):
    if rescale:
        dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255)
    else:
        dataGenerator = tf.keras.preprocessing.image.ImageDataGenerator()

    tf_dataset = dataGenerator.flow_from_dataframe(
        dataframe, directory=directory_path, x_col='path', y_col='minifigure_name', target_size=(img_size, img_size),
        batch_size=batch_size, )
    return tf_dataset


trainDataset = tf_load_data(TRAIN_DATA)
testDataset = tf_load_data(TEST_DATA)


def tensorLogging(dir, testName):  # logs tensorflow related actions
    logDir = dir + '/' + testName + '/' + datetime.datetime.now().strftime("%m%d%Y%H%M%S")
    tensorLog = tf.keras.callbacks.TensorBoard(log_dir=logDir)
    print(f"Logging has started and will be outputted to: {logDir}")
    return tensorLog


checkpoint = 'modelCheckpoint/mlp.ckpt'

# This generates a massive 900mb folder. Delete it before you push to github
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True,
                                                     save_weights_only=True, verbose=0)


def plotLossCurves(history):
    loss = history.history['loss']
    valLoss = history.history['val_loss']

    accuracy = history.history['accuracy']
    valAccuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='trainingLoss')  # training loss
    plt.plot(epochs, valLoss, label='valLoss')  # validation loss
    plt.title('Loss with a seed of '+str(seed))
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracy, label='trainingAccuracy')  # training accuracy
    plt.plot(epochs, valAccuracy, label='valAccuracy')  # validation accuracy
    plt.title('Accuracy with a seed of '+str(seed))
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

# Convolutional network:
print("=========================================================================================================="
      "\n\nCONVOLUTIONAL NETWORK BEGINS HERE!\n\n==========================================================================================================")


tf.random.set_seed(seed)  # takes a random seed

inputs = tf.keras.layers.Input(shape=trainDataset.image_shape, name='input')

x = tf.keras.layers.Conv2D(filters=10, activation='relu', kernel_size=3, name='conv_1_1')(inputs)
x = tf.keras.layers.Conv2D(filters=10, activation='relu', kernel_size=3, name='conv_1_2')(x)
x = tf.keras.layers.MaxPool2D(pool_size=2, padding='valid', name='max_pool_1')(x)

x = tf.keras.layers.Conv2D(filters=10, activation='relu', kernel_size=3, name='conv_2_1')(x)
x = tf.keras.layers.Conv2D(filters=10, activation='relu', kernel_size=3, name='conv_2_2')(x)
x = tf.keras.layers.MaxPool2D(pool_size=2, padding='valid', name='max_pool_2')(x)

x = tf.keras.layers.Flatten(name='flatten_layer')(x)
outputs = tf.keras.layers.Dense(36, activation='softmax', name='output_layer')(x)

model_tiny_vgg = tf.keras.Model(inputs, outputs)

model_tiny_vgg.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
tf.keras.utils.plot_model(model_tiny_vgg, "model.png", show_shapes=True, show_layer_names=True)

model_tiny_vgg.summary()

history_tiny_vgg = model_tiny_vgg.fit(trainDataset,
                                      epochs=36,
                                      steps_per_epoch=len(trainDataset),
                                      validation_data=testDataset,
                                      validation_steps=len(testDataset),
                                      callbacks=[tensorLogging('training_logs', 'lego_tiny_vgg'),
                                                 modelCheckpoint])

results_tiny_vgg = model_tiny_vgg.evaluate(testDataset)
print(results_tiny_vgg)

plotLossCurves(history_tiny_vgg)  # plots the performance

# additional test images (not all square) add a lot of variance in accuracy in the final model
