from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS

original_dim = RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


class Sampling(layers.Layer):

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim),
                                                 mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def call(self, latent_dim, inputs, **kwargs):
        inputs = keras.Input(shape=(original_dim,))
        x = Dense(512, activation='relu')(inputs)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        # encoder.summary()

        return encoder


class Decoder(layers.Layer):

    def call(self, latent_dim, latent_inputs, **kwargs):
        latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(512,
                  activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l1(0.001))(latent_inputs)
        decoder_outputs = Dense(original_dim, activation='sigmoid')(x)

        decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
        # decoder.summary()

        return decoder


class VAE(keras.Model, ABC):
    '''
    Define the VAE as a `Model` with a custom `train_step`
    '''

    def __init__(self, model_name, loss, latent_dim, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.model_name = model_name
        self.intermediate_dim = 512
        self.latent_dim = latent_dim
        self.lossFunc = loss
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
            reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

            if self.lossFunc == "VAE":
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_mean(kl_loss)
                kl_loss *= -0.5
                total_loss = reconstruction_loss + kl_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }
            else:
                total_loss = reconstruction_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                }

    def call(self, inputs, **kwargs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(inputs, reconstruction))
        reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

        if self.lossFunc[0] == "VAE":
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            self.add_metric(total_loss, name='total_loss', aggregation='mean')
            self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
            return reconstruction
        else:
            total_loss = reconstruction_loss
            self.add_metric(total_loss, name='total_loss', aggregation='mean')
            self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
            return reconstruction


def get_input_shape():
    return (original_dim,)


def get_image_dim():
    return RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


def normalize_and_reshape(x):
    x = x.astype('float32') / 255.
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x


def reshape(x):
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x
