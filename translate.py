import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def load_data(city_name):
    xs = np.load('data//pre-processed//'+ city_name + '_finalized_x.npy')
    xs = np.expand_dims(xs, axis=1)
    ys = np.load('data//pre-processed//'+ city_name + '_finalized_y.npy')
    print("{} xs has shape {}".format(city_name, xs.shape))
    print("{} ys has shape {}".format(city_name, ys.shape))
    return xs, ys

beijing_xs, beijing_ys = load_data('beijing')
tianjin_xs, tianjin_ys = load_data('tianjin')
shenzhen_xs, shenzhen_ys = load_data('shenzhen')
guangzhou_xs, guangzhou_ys = load_data('guangzhou')

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(500, use_bias=False, input_shape = (1, 18)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(128, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(64, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv1DTranspose(1, 5, strides=1, padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 1, strides=1, padding='same',
                                     input_shape = (1, 18)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
discriminator.predict(beijing_xs)
