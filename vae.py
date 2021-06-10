#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.metrics import Mean
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Layer, Reshape
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.losses import binary_crossentropy

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
    return -0.5 * tfm.reduce_mean(tfm.reduce_sum(difference, axis=1))

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
        self.convs = [(Conv2D(64/(i+1), 3, padding='same'), BatchNorm())
                      for i in range(conv_amt)]
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
        self.transpose_convs = [(Conv2DTranspose(64/(i+1), 3, padding='same'), BatchNorm())
                                for i in reversed(range(conv_amt))]
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
        return sigmoid(self.output_layer(transpose_conv_output))

class VAE(Model):
    
    def __init__(self, image_shape, conv_amt=5, dense_amt=5,
                 regularization_weight=1e-3, middle_dim=256, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.encoder, self.decoder = self.create_layers(conv_amt, dense_amt, middle_dim)
        self.regularization_weight = regularization_weight
        self.loss_tracker = Mean(name='loss')
        self.reconstruction_tracker = Mean(name='reconstruction_loss')
        self.kl_tracker = Mean(name='kl_tracker')
        self.sampling = Sampling()

    def create_layers(self, conv_amt, dense_amt, middle_dim):
        enc_conv = dec_conv = conv_amt
        enc_dense = dec_dense = dense_amt
        if isinstance(conv_amt, tuple):
            enc_conv, dec_conv = conv_amt
        if isinstance(dense_amt, tuple):
            enc_dense, dec_dense = dense_amt
        encoder = Encoder(
            conv_amt=enc_conv,
            dense_amt=enc_dense,
            latent_dim=self.latent_dim,
            middle_dim=middle_dim
        )
        decoder = Decoder(
            self.image_shape,
            conv_amt=dec_conv,
            dense_amt=dec_dense,
            middle_dim=middle_dim
        )
        return encoder, decoder

        
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
            bce = tfm.reduce_mean(
                tfm.reduce_sum(
                    binary_crossentropy(data, reconstruction), 
                    axis=(1, 2)
                )
            )
            kl = sum(self.losses)
            loss = bce + self.regularization_weight * kl
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.reconstruction_tracker.update_state(bce)
        self.kl_tracker.update_state(kl)
        pixel_count = reconstruction.shape[1] * reconstruction.shape[2]
        mean_reconstruction_loss = self.reconstruction_tracker.result()/pixel_count
        mean_kl_loss = self.kl_tracker.result()/self.latent_dim
        return {
            "loss": self.loss_tracker.result(), 
            "reconstruction_loss": self.reconstruction_tracker.result(),
            "kl_loss": self.kl_tracker.result(),
            "mean_reconstruction_loss": mean_reconstruction_loss,
            "mean_kl_loss": mean_kl_loss
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]
