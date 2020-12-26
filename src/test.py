
import keras.backend as K
from keras.layers import Input, Dense, concatenate, Conv2D, Reshape, Flatten
from keras.losses import mse
from keras import Model
from utils.ObjectUtil import ObjectUtil
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2])
print(a * b.reshape((2, 1)))