import os
import numpy as np
import pretty_errors
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
    model.add(layers.Conv1D(64, 1, strides=1, padding='same',
                                     input_shape = (1, 18)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv1D(128, 1, strides=1, padding='same'))
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
generator = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train_step(images):
    noise = np.asarray([np.random.rand(1, 18)])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

train_step(beijing_xs)
