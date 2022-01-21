import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

def load_data(city_name):
    xs = np.load('data//pre-processed//'+ city_name + '_finalized_x.npy')
    ys = np.load('data//pre-processed//'+ city_name + '_finalized_y.npy')
    print("{} xs has shape {}".format(city_name, xs.shape))
    print("{} ys has shape {}".format(city_name, ys.shape))
    return xs, ys

beijing_xs, beijing_ys = load_data('beijing')
tianjin_xs, tianjin_ys = load_data('tianjin')
shenzhen_xs, shenzhen_ys = load_data('shenzhen')
guangzhou_xs, guangzhou_ys = load_data('guangzhou')

X_train = beijing_xs
y_train = beijing_ys

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (1, 18)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
