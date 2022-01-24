import os
import numpy as np
import pretty_errors
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
tianjin_xs, tianjin_ys = load_data('tianjin')
shenzhen_xs, shenzhen_ys = load_data('shenzhen')
guangzhou_xs, guangzhou_ys = load_data('guangzhou')

tianjin_to_beijing = np.load('translated//tianjin_to_beijing.npy')
shenzhen_to_beijing = np.load('translated//shenzhen_to_beijing.npy')
guangzhou_to_beijing = np.load('translated//guangzhou_to_beijing.npy')

tianjin_to_beijing_no_m = np.load('translated//tianjin_to_beijing_no_mahalanobis.npy')
shenzhen_to_beijing_no_m = np.load('translated//shenzhen_to_beijing_no_mahalanobis.npy')
guangzhou_to_beijing_no_m = np.load('translated//guangzhou_to_beijing_no_mahalanobis.npy')

X_train = beijing_xs
y_train = beijing_ys

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

regressor = Sequential()
regressor.add(LSTM(units = 2048, return_sequences = True, input_shape = (1, 18)))
regressor.add(Dropout(0.2))
#regressor.add(LSTM(units = 512, return_sequences = True))
#regressor.add(Dropout(0.2))
#regressor.add(LSTM(units = 1024, return_sequences = True))
#regressor.add(Dropout(0.2))
#regressor.add(LSTM(units = 512, return_sequences = True))
#regressor.add(Dropout(0.2))
#regressor.add(LSTM(units = 256))
#regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 1500, batch_size = 32)

y_pred = np.squeeze(regressor.predict(X_test))
mse = mean_squared_error(y_test, y_pred)
print('the testing mse error is {}'.format(mse))

y_pred_tianjin = np.squeeze(regressor.predict(np.expand_dims(tianjin_to_beijing, axis=1)))
mse_t_b = mean_squared_error(tianjin_ys, y_pred_tianjin)
print('the testing mse of translated tianjin is {}'.format(mse_t_b))

y_pred_shenzhen = np.squeeze(regressor.predict(np.expand_dims(shenzhen_to_beijing, axis=1)))
mse_s_b = mean_squared_error(shenzhen_ys, y_pred_shenzhen)
print('the testing mse of translated shenzhen is {}'.format(mse_s_b))

y_pred_guangzhou = np.squeeze(regressor.predict(np.expand_dims(guangzhou_to_beijing, axis=1)))
mse_g_b = mean_squared_error(guangzhou_ys, y_pred_guangzhou)
print('the testing mse of translated guangzhou is {}'.format(mse_g_b))

print('#######################################################################')

y_pred_tianjin = np.squeeze(regressor.predict(np.expand_dims(tianjin_to_beijing_no_m, axis=1)))
mse_t_b = mean_squared_error(tianjin_ys, y_pred_tianjin)
print('the testing mse of translated tianjin w/o mahalanobis is {}'.format(mse_t_b))

y_pred_shenzhen = np.squeeze(regressor.predict(np.expand_dims(shenzhen_to_beijing_no_m, axis=1)))
mse_s_b = mean_squared_error(shenzhen_ys, y_pred_shenzhen)
print('the testing mse of translated shenzhen w/o mahalanobis is {}'.format(mse_s_b))

y_pred_guangzhou = np.squeeze(regressor.predict(np.expand_dims(guangzhou_to_beijing_no_m, axis=1)))
mse_g_b = mean_squared_error(guangzhou_ys, y_pred_guangzhou)
print('the testing mse of translated guangzhou w/o mahalanobis is {}'.format(mse_g_b))
