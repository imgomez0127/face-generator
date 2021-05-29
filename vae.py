#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Layer, Reshape
from tensorflow.keras.layers import BatchNormalization as BatchNorm
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
    difference = 1 + sigma_log - tfm.square(mu) - tfm.exp(sigma_log)
    return -0.5 * tfm.reduce_mean(difference)

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
        self.convs = [(Conv2D(3, 3, padding='same'), BatchNorm())
                      for _ in range(conv_amt)]
        self.dense_layers = [(Dense(middle_dim), BatchNorm())
                             for _ in range(dense_amt)]
        self.flatten = Flatten()
        self.latent_mu = Dense(latent_dim)
        self.latent_sig = Dense(latent_dim)

    def call(self, inputs):
        conv_output = inputs
        for conv_layer, norm in self.convs:
            conv_output = relu(norm(conv_layer(conv_output)))
        intermediate_output = self.flatten(conv_output)
        for dense, norm in self.dense_layers:
            intermediate_output = relu(norm(dense(intermediate_output)))
        mu = self.latent_mu(intermediate_output)
        sigma_log = self.latent_sig(intermediate_output)
        return mu, sigma_log

class Decoder(Layer):

    def __init__(self, image_shape, conv_amt=5, dense_amt=5, middle_dim=256):
        super().__init__()
        flatten_shape = int(tfm.reduce_prod(image_shape).numpy())
        self.transpose_convs = [(Conv2DTranspose(3, 3, padding='same'), BatchNorm())
                                for _ in range(conv_amt)]
        self.output_layer = Conv2DTranspose(3, 3, padding='same')
        self.dense_layers = [(Dense(middle_dim), BatchNorm())
                             for _ in range(dense_amt)]
        self.dense2 = Dense(flatten_shape)
        self.reshape = Reshape(image_shape)

    def call(self, inputs):
        for dense, norm in self.dense_layers:
            inputs = relu(norm(dense(inputs)))
        dense_output2 = relu(self.dense2(inputs))
        transpose_conv_output = self.reshape(dense_output2)
        for transpose_conv, norm in reversed(self.transpose_convs):
            transpose_conv_output = norm(transpose_conv(transpose_conv_output))
            transpose_conv_output = relu(transpose_conv_output)
        return relu(self.output_layer(transpose_conv_output))

class VAE(Model):

    def __init__(self, image_shape, conv_amt=5, dense_amt=5,
                 regularization_weight=1e-3, middle_dim=256, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.encoder = Encoder(
            conv_amt=conv_amt,
            dense_amt=dense_amt,
            latent_dim=latent_dim,
            middle_dim=middle_dim
        )
        self.decoder = Decoder(
            image_shape,
            conv_amt=conv_amt,
            dense_amt=dense_amt,
            middle_dim=middle_dim
        )
        self.regularization_weight = regularization_weight
        self.loss_tracker = Mean(name='loss')
        self.bce_loss = BinaryCrossentropy()
        self.sampling = Sampling()

    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        self.add_loss(kl_loss(*encoder_output))
        sampled_input = self.sampling(encoder_output)
        return self.decoder(sampled_input)

    def generate_image(self, sigma_log=None, mu=None):
        if mu is None:
            mu = tf.random.uniform(shape=[1, self.latent_dim])
        if sigma_log is None:
            sigma_log = tf.random.uniform(shape=[1, self.latent_dim])
        inputs = (mu, sigma_log)
        sampled_vector = self.sampling(inputs)
        return self.decoder(sampled_vector)

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
