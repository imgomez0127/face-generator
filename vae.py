#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Layer, Reshape, Input
from tensorflow.keras.losses import BinaryCrossentropy

class Sampling(Layer):

    def call(self, inputs):
        mu, sigma_log = inputs
        batches, width = sigma_log.shape
        eps = tf.random.normal((batches, width))
        sigma = tfm.exp(sigma_log * 0.5)
        sigma = sigma * eps
        return mu + sigma


def kl_loss(mu, sigma_log):
    return -0.5 * tfm.reduce_mean(1 + sigma_log - tfm.square(mu) - tfm.exp(sigma_log))

def compute_conv_shape(image_shape, conv_amt, conv_size=None):
    if not conv_size:
        conv_size = (3, 3)
    height, width, *rest = image_shape
    for _ in range(conv_amt):
        height -= conv_size[0]-1
        width -= conv_size[1]-1
    return (height, width, *rest)

class Encoder(Layer):

    def __init__(self, conv_amt=5, dense_amt=5, middle_dim=256, latent_dim=128):
        super().__init__()
        self.convs = [Conv2D(3, 3, padding='same', activation='relu')
                      for _ in range(conv_amt)]
        self.dense_layers = [Dense(middle_dim, activation='relu')
                             for _ in range(dense_amt)]
        self.flatten = Flatten()
        self.latent_mu = Dense(latent_dim)
        self.latent_sig = Dense(latent_dim)

    def call(self, inputs):
        conv_output = inputs
        for conv_layer in self.convs:
            conv_output = conv_layer(conv_output)
        intermediate_output = self.flatten(conv_output)
        for dense in self.dense_layers:
            intermediate_output = dense(intermediate_output)
        mu = self.latent_mu(intermediate_output)
        log_sigma = self.latent_sig(intermediate_output)
        return mu, log_sigma

class Decoder(Layer):

    def __init__(self, image_shape, conv_amt=5, dense_amt=5, middle_dim=256):
        super().__init__()
        flatten_shape = int(tfm.reduce_prod(image_shape).numpy())
        self.transpose_convs = [Conv2DTranspose(3, 3, padding='same', activation="relu")
                                for _ in range(conv_amt-1)]
        output_layer = [Conv2DTranspose(3, 3, padding='same')]
        self.transpose_convs = output_layer + self.transpose_convs
        self.dense_layers = [Dense(middle_dim, activation='relu')
                             for _ in range(dense_amt)]
        self.dense2 = Dense(flatten_shape, activation='relu')
        self.reshape = Reshape(image_shape)

    def call(self, inputs):
        for dense in self.dense_layers:
            inputs = dense(inputs)
        dense_output2 = self.dense2(inputs)
        transpose_conv_output = self.reshape(dense_output2)
        for transpose_conv in reversed(self.transpose_convs):
            transpose_conv_output = transpose_conv(transpose_conv_output)
        return transpose_conv_output

class VAE(Model):

    def __init__(self, image_shape, conv_amt=5, regularization_weight=1e-3):
        super().__init__()
        self.encoder = Encoder(conv_amt=conv_amt)
        self.decoder = Decoder(image_shape, conv_amt=conv_amt)
        self.regularization_weight = regularization_weight
        self.loss_tracker = Mean(name='loss')
        self.bce_loss = BinaryCrossentropy()
        self.sampling = Sampling()

    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        self.add_loss(kl_loss(*encoder_output))
        sampled_input = self.sampling(encoder_output)
        return self.decoder(sampled_input)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self(data, training=True)
            loss = self.bce_loss(data, reconstruction)
            loss += self.regularization_weight * sum(self.losses)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
