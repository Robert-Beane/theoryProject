#  start of main training
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import datetime
import tensorflow_hub as hub

#  Setting up static seed
static_seed = 420
np.seed(static_seed)
tf.set_seed(static_seed)

#  Setting up randomized seed
# ran_seed = 420
# random.seed(ran_seed)
# np.random.seed(ran_seed)
# tf.random.set_seed(ran_seed)


TRAIN_DATA = pd.read_csv('index.csv')
TEST_DATA = pd.read_csv('test.csv')
METADATA = pd.read_csv('metadata.csv')