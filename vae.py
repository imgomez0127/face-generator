#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2d, Conv2DTranspose, Flatten, Dense, Layer, Reshape

class Sampling(Layer):

    def call(self, inputs):
        mu, sigma_log = inputs
        eps = tf.random.normal(mu.shape)
        sigma = tfm.exp(sigma_log * 0.5)
        sigma = sigma * eps
        return mu + sigma


def kl_loss(mu, sigma_log):
    return -0.5 * tfm.reduce_mean(1 + sigma_log - tfm.square(mu) - tfm.exp(sigma_log), axis=-1)

def compute_conv_shape(image_shape, conv_amt, conv_size=None):
    if not conv_size:
        conv_size = (3, 3)
    height, width, *rest = image_shape
    for _ in range(conv_amt):
        height -= conv_size-1
        width -= conv_size-1
    return (height, width, *rest)

class Encoder(Layer):

    def __init__(self, conv_amt=3, middle_dim=256, latent_dim=128):
        super().__init__()
        self.convs = [Conv2d(3, 3, padding='same') for _ in range(conv_amt)]
        self.dense = Dense(middle_dim)
        self.flatten = Flatten()
        self.latent_mu = Dense(latent_dim)
        self.latent_sig = Dense(latent_dim)

    def call(self, inputs):
        conv_output = inputs
        for conv_layer in self.convs:
            conv_output = conv_layer(conv_output)
        flatten_output = self.flatten(conv_output)
        intermediate_output = self.dense(flatten_output)
        mu = self.latent_mu(intermediate_output)
        log_sigma = self.latent_sig(intermediate_output)
        return mu, log_sigma

class Decoder(Layer):

    def __init__(self, image_shape, conv_amt=3, middle_dim=256):
        super().__init__()
        flatten_shape = int(tfm.reduce_prod(image_shape).numpy())
        self.transpose_convs = [Conv2DTranspose(3, 3, padding='same') for _ in range(conv_amt)]
        self.dense = Dense(middle_dim)
        self.dense2 = Dense(flatten_shape)
        self.reshape = Reshape(image_shape)
        self.sampling = Sampling()

    def call(self, inputs):
        sampled_vector = self.sampling(inputs)
        dense_output = self.dense(sampled_vector)
        dense_output2 = self.dense2(dense_output)
        transpose_conv_output = self.reshape(dense_output2)
        for transpose_conv in reversed(self.transpose_convs):
            transpose_conv_output = transpose_conv(dense_output2)
        return transpose_conv_output

class VAE(Model):

    def __init__(self, image_shape, conv_amt=3, regularization_weight=0.001):
        super().__init__()
        self.post_conv_shape = compute_conv_shape(image_shape, conv_amt)
        self.encoder = Encoder(conv_amt=3)
        self.decoder = Decoder(image_shape, conv_amt=3)
        self.regularization_weight = regularization_weight

    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        self.add_loss(kl_loss(*encoder_output))
        return self.decoder(encoder_output)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction = self(data, training=True)
            loss = self.compile_loss(data, reconstruction)
            loss += self.regularization_weight * sum(self.losses)
        trainable_vars = self.trainable_vars
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(data, reconstruction)
        return {m.name: m.result() for m in self.metrics}
