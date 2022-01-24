import os
import numpy as np
import pretty_errors
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mahalanobis import inv_empirical_cov
from mahalanobis import empirical_mean

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
    model.add(layers.Conv1DTranspose(64, 1, strides=1, padding='same',
                                     input_shape = (1, 18)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1DTranspose(128, 2, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1DTranspose(256, 2, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1DTranspose(128, 1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(18))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, 1, strides=1, padding='same',
                                     input_shape = (1, 18)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 1, strides=1, padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
generator_a_b = make_generator_model()
generator_b_a = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

LAMBDA = 0.01


def generator_loss(real_output, fake_output):
    #return cross_entropy(real_output, fake_output)
    return fake_output

generator_a_b_optimizer = tf.keras.optimizers.Adam(1e-5)
generator_b_a_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

BATCH = 1
def train_step(images, target_images):
    #noise = np.random.rand(BATCH, 1, 18)
    noise = np.asarray([random.choice(target_images)])
    with tf.GradientTape() as gen_tape_a_b, tf.GradientTape() as gen_tape_b_a, tf.GradientTape() as disc_tape:
        generated_images = tf.expand_dims(generator_a_b(noise, training=True), axis=1)
        generated_back_images = tf.expand_dims(generator_b_a(generated_images, training=True), axis=1)

        real_output = discriminator(np.asarray([images]), training=True)

        fake_output = discriminator(generated_images, training=True)


        gen_loss = generator_loss(generated_images)
        disc_loss = discriminator_loss(real_output, fake_output)

    print('generator loss: {}, discriminator_loss:{}'.format(gen_loss, disc_loss))
    gradients_of_generator_a_b = gen_tape_a_b.gradient(gen_loss, generator_a_b.trainable_variables)
    gradients_of_generator_b_a = gen_tape_b_a.gradient(gen_loss, generator_b_a.trainable_variables)


    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_a_b_optimizer.apply_gradients(zip(gradients_of_generator_a_b, generator_a_b.trainable_variables))
    generator_b_a_optimizer.apply_gradients(zip(gradients_of_generator_b_a, generator_b_a.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, target_images, epochs):
  for epoch in range(epochs):
      print('training the {}th epoch'.format(epoch))
      print('--------------------------------------')
      for image_batch in dataset:
          train_step(image_batch, target_images)

try:
    os.mkdir('.//translated')
except:
    pass

# beijing to tianjin
train(beijing_xs, tianjin_xs, 10)
b_to_t = generator(tianjin_xs)
np.save('.//translated//tianjin_to_beijing_disco.npy', b_to_t)

train(beijing_xs, shenzhen_xs, 10)
b_to_s = generator(shenzhen_xs)
np.save('.//translated//shenzhen_to_beijing_disco.npy', b_to_s)

train(beijing_xs, guangzhou_xs, 10)
b_to_g = generator(guangzhou_xs)
np.save('.//translated//guangzhou_to_beijing_disco.npy', b_to_g)
