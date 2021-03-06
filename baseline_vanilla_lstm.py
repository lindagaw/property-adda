import os
import numpy as np
import pretty_errors
import tensorflow as tf
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from error_rate import error_rate

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def load_air_quality_data(city):
    xs = np.load('data//pre-processed//' + city + '_pm25_xs.npy')
    ys = np.load('data//pre-processed//' + city + '_pm25_ys.npy')

    xs = np.expand_dims(xs, axis=1)

    return xs, ys


X, y = load_air_quality_data(sys.argv[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regressor = Sequential()
regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (1, 19)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 1024, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 2048, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 1024, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 256))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
, loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 500, batch_size = 32)

y_pred = np.squeeze(regressor.predict(X_test))

mse = mean_squared_error(y_test, y_pred, squared=True)
accuracy = error_rate(y_test, y_pred, 5)

print('the testing mse error is {}'.format(mse))
print('the accuracy is {}'.format(accuracy))
