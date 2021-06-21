#!/usr/bin/env python3
import math
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers

def _product(*args):
    prod = 1
    for arg in args:
        prod *= arg
    return prod


def _get_discriminator_targets(targets, smoothing=True):
    if smoothing:
        targets = tf.random.uniform(targets.shape, minval=0.9) * targets
        negative_labels = tf.random.uniform(targets.shape, maxval=0.1)
    else:
        negative_labels = tf.zeros(targets.shape)
    return tf.concat([targets, negative_labels], 0)


def _perturb_images(src, stddev=0.5):
    return (src + tf.random.normal(src.shape, stddev=stddev))/(1+stddev)

class Generator(layers.Layer):

    def __init__(self, latent_shape, conv_params=None):
        super().__init__()
        if not conv_params:
            raise ValueError("Did not add any input conv_params")
        self.batch_norms = [layers.BatchNormalization()
                            for _ in range(len(conv_params)-1)]
        self.conv_layers = [layers.Conv2DTranspose(*conv_args, **conv_kwargs)
                            for conv_args, conv_kwargs in conv_params[:-1]]
        self.latent_shape = latent_shape
        self.reshape_layer = layers.Reshape(_product(self.latent_shape))
        self.output_layer = layers.Conv2DTranspose(*conv_params[-1][0],
                                                   **conv_params[-1][1])

    def call(self, inputs):
        # Get input data from the dataset
        samples, _ = inputs
        samples = self.reshape_layer(samples)
        for conv, norm in zip(self.conv_layers, self.batch_norms):
            samples = activations.relu(norm(conv(samples)))
        return activations.sigmoid(self.output_layer(samples))


class Discriminator(layers.Layer):

    def __init__(self, conv_params=None, dense_params=None):
        super().__init__()
        if not (conv_params and dense_params):
            raise ValueError("Did not add any parameters for either conv_params or dense_params")
        dropout_amt = len(conv_params)+len(dense_params)-1
        self.dropout_layers = [layers.Dropout(0.5)
                               for _ in range(dropout_amt)]
        self.conv_layers = [layers.Conv2D(*conv_args, **conv_kwargs)
                            for conv_args, conv_kwargs in conv_params]
        self.dense_layers = [layers.Dense(*dense_args, **dense_kwargs)
                             for dense_args, dense_kwargs in dense_params[:-1]]
        self.output_layer = layers.Dense(*dense_params[-1][0],
                                         **dense_params[-1][1])
        self.leaky_relu = layers.LeakyReLU()
        self.flatten = layers.Flatten()

    def call(self, inputs):
        data, _ = inputs
        conv_dropouts = self.dropout_layers[:len(self.conv_layers)]
        dense_dropouts = self.dropout_layers[len(self.conv_layers):]
        for conv, dropout in zip(self.conv_layers, conv_dropouts):
            data = self.leaky_relu(conv(data))
            data = dropout(data)
        data = self.flatten(data)
        for dense, dropout, norm in zip(self.dense_layers, dense_dropouts):
            data = self.leaky_relu(dense(data))
            data = dropout(data)
        return activations.sigmoid(self.output_layer(data))


class GAN(keras.Model):

    def __init__(self, output_dim, discriminator_params, generator_params):
        super().__init__()
        self.output_dim = output_dim
        self.discriminator = Discriminator(*discriminator_params[0],
                                           **discriminator_params[1])
        self.generator = Generator(*generator_params[0], **generator_params[1])
        self.bce = losses.BinaryCrossentropy()
        self.disc_loss_tracker = metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = metrics.Mean(name="gen_loss")
        self.disc_optimizer = optimizers.Adam(learning_rate=5e-5, beta_1=0.5)
        self.latent_len = _product(*self.generator.latent_shape)


    def call(self, inputs):
        return self.generator(inputs)

    def train_discriminator(self, inputs):
        _, targets = inputs
        predictions = self.discriminator(inputs)
        return self.bce(targets, predictions)

    def train_generator(self, inputs):
        _, targets = inputs
        predictions = self.discriminator((self(inputs), targets))
        return self.bce(targets, predictions)

    def train_step(self, inputs):
        src, targets = inputs, tf.ones((inputs.shape[0], 1))
        sampled_vector = tf.random.uniform((src.shape[0], self.latent_len), minval=-1, maxval=1)
        fake_images = self((sampled_vector, None))
        training_imgs = tf.concat([src, fake_images], 0)
        training_imgs = _perturb_images(training_imgs, stddev=10/255)
        targets = _get_discriminator_targets(targets)
        self.generator.trainable = False
        self.discriminator.trainable = True
        with tf.GradientTape() as disc_tape:
            disc_loss = self.train_discriminator((training_imgs, targets))
        disc_gradients = disc_tape.gradient(disc_loss, self.trainable_weights)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.trainable_weights))
        self.disc_loss_tracker.update_state(disc_loss)
        self.discriminator.trainable = False
        self.generator.trainable = True
        sampled_vectors = tf.random.uniform((src.shape[0], self.latent_len), minval=-1, maxval=1)
        gen_targets = tf.ones((src.shape[0], 1))
        with tf.GradientTape() as gen_tape:
            gen_loss = self.train_generator((sampled_vectors, gen_targets))
        gen_gradients = gen_tape.gradient(gen_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gen_gradients, self.trainable_weights))
        self.gen_loss_tracker.update_state(gen_loss)
        return {
            "disc_loss": self.disc_loss_tracker.result(),
            "gen_loss": self.gen_loss_tracker.result()
        }

    @property
    def metrics(self):
        return [self.disc_loss_tracker, self.gen_loss_tracker]
