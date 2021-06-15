#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras import losses

def product(*args):
    prod = 1
    for arg in args:
        prod *= arg
    return prod

class Generator(layers.Layer):

    def __init__(self, output_shape, conv_params=None):
        if not conv_params:
            raise ValueError("Did not add any input conv_params")
        self.latent_dim = (output_shape[0]//len(conv_params),
                           output_shape[1]//len(conv_params),
                           output_shape[2])
        self.batch_norms = [layers.BatchNormalization(momentum=0.9)
                            for _ in range(len(conv_args))]
        self.upsampling_layers = [layers.UpSampling2D()
                                  for _ in range(len(conv_params)-1)]
        self.conv_layers = [layers.Conv2DTranspose(*conv_arg, **conv_kwarg)
                            for conv_args, conv_kwargs in conv_params[:-1]]
        self.sample = layers.Dense(product(*self.latent_dim))
        self.reshape_layer = layers.Reshape(self.latent_dim)
        self.output_layer = layers.Conv2DTranspose(*conv_params[-1][0],
                                                   **conv_params[-1][1])

    def call(self, inputs):
        # Get input data from the dataset
        data, _ = inputs
        samples = activations.relu(self.batch_norms[0](self.sample(data)))
        for upsample, conv, norm in zip(self.upsampling_layers,
                                        self.conv_layers, self.batch_norms[1:]):
            samples = upsample(samples)
            samples = norm(activations.relu(conv(samples)))
        return activations.sigmoid(self.output_layer(samples))


class Discriminator(layers.Layer):

    def __init__(self, conv_params=None, dense_params=None):
        if not (conv_params and dense_params):
            raise ValueError("Did not add any parameters for either conv_params or dense_params")
        dropout_amt = len(conv_params)+len(dense_params)-1
        self.dropout_layers = [layers.Dropout(0.5)
                               for _ in range(dropout_amt)]
        self.conv_layers = [layers.Conv2D(*conv_arg, **conv_kwarg)
                            for conv_args, conv_kwargs in conv_params]
        self.dense_layers = [layers.Dense(*dense_args, **dense_kwargs)
                             for dense_args, dense_kwargs in dense_params[:-1]]
        self.output_layer = layers.Dense(*dense_params[-1][0],
                                         **dense_params[-1][1])
        self.leaky_relu = layers.LeakyReLU()


    def call(self, inputs):
        data, target = inputs
        conv_dropouts = self.dropout_layers[:len(self.conv_layers)]
        dense_dropouts = self.dropout_layers[len(self.conv_layers):]
        for conv, dropout in zip(self.conv_layers, self.dropout_layers):
            data = self.leaky_relu(self.conv(data))
            data = dropout(data)
        for dense, dropout in zip(self.dense_layers, dense_dropouts):
            data = self.leaky_relu(dense(data))
            data = dropout(data)
        return activations.sigmoid(self.output_layer(data))
