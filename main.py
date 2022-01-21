import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

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
#tianjin_xs, tianjin_ys = load_data('tianjin')
#shenzhen_xs, shenzhen_ys = load_data('shenzhen')
#guangzhou_xs, guangzhou_ys = load_data('guangzhou')

X_train = beijing_xs
y_train = beijing_ys

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

regressor = Sequential()
for i in range(0, 8):
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)

y_pred = np.squeeze(regressor.predict(X_test))

mse = mean_squared_error(y_test, y_pred)

print('the testing mse error is {}'.format(mse))
