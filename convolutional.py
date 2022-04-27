#  start of main training
import pandas as pd
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import datetime

# import tensorflow_hub as hub

#  Setting up static seed
# static_seed = 420
# np.seed(static_seed)
# tf.set_seed(static_seed)

#  Setting up randomized seed
seed = 420
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

PATH = '../theoryProject/'
IMG_SIZE = 512


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

#  begin MLP (multilayer perception) model

tf.random.set_seed(seed)  # seed is 420

model_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(trainDataset.image_shape), name='input'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(100, activation='relu', name='first_hidden_layer'),
    tf.keras.layers.Dense(64, activation='relu', name='second_hidden_layer'),
    tf.keras.layers.Dense(36, activation='softmax', name='output_layer')
], name='MLP')

model_mlp.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

model_mlp.summary()


# toyTensor = tf.cast(tf.range(-10, 10), tf.float32)
# print(toyTensor)
# plt.plot(toyTensor)
# plt.show()


def tensorLogging(dir, testName):
    logDir = dir + '/' + testName + '/' + datetime.datetime.now().strftime("%m%d%Y%H%M%S")
    tensorLog = tf.keras.callbacks.TensorBoard(log_dir=logDir)
    print(f"Logging has started and will be outputted to: {logDir}")
    return tensorLog


checkpoint = 'modelCheckpoint/mlp.ckpt'

# This generates a massive 900mb folder. Delete it before you push to github
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True,
                                                     save_weights_only=True, verbose=0)

#  time to train
mlpTrain = model_mlp.fit(trainDataset, epochs=12, steps_per_epoch=len(trainDataset), validation_data=testDataset,
                         validation_steps=len(testDataset),
                         callbacks=[tensorLogging('training_logs', 'lego_mlp'), modelCheckpoint])

results = model_mlp.evaluate(testDataset)
print(results)


#  start plotting MLP performance


def plotLossCurves(history):
    loss = history.history['loss']
    valLoss = history.history['val_loss']

    accuracy = history.history['accuracy']
    valAccuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='trainingLoss')
    plt.plot(epochs, valLoss, label='valLoss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracy, label='trainingAccuracy')
    plt.plot(epochs, valAccuracy, label='valAccuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


plotLossCurves(mlpTrain)

# Convolutional network:
print("==========================================================================================================")
print("")
print("CONVOLUTIONAL NETWORK BEGINS HERE!")
print("")
print("==========================================================================================================")

tf.random.set_seed(666)

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

tf.random.set_seed(563)

history_tiny_vgg = model_tiny_vgg.fit(TRAIN_DATA,
                                      epochs=12,
                                      steps_per_epoch=len(TRAIN_DATA),
                                      validation_data=TEST_DATA,
                                      validation_steps=len(TEST_DATA),
                                      callbacks=[tensorLogging('training_logs', 'lego_tiny_vgg'),
                                                 modelCheckpoint])

results_tiny_vgg = model_tiny_vgg.evaluate(TEST_DATA)
print(results_tiny_vgg)

plotLossCurves(history_tiny_vgg)

# additional test images (not all square) add a lot of variance in accuracy in the final model
