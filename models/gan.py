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
        self.batch_norms = [layers.BatchNormalization(momentum=0.9)
                            for _ in range(len(conv_args)+1)]
        self.conv_layers = [layers.Conv2DTranspose(*conv_arg, **conv_kwarg)
                            for conv_args, conv_kwargs in conv_params]
        self.sample = layers.Dense(product(*output_shape))


class Discriminator(layers.Layer):

    def __init__(self, conv_params=None, dense_params=None):
        if not (conv_params and dense_params):
            raise ValueError("Did not add any parameters for either conv_params or dense_params")
        self.batch_norms = [layers.BatchNormalization()
                            for _ in range(len(conv_params)+len(dense_params))]
        self.conv_layers = [layers.Conv2DTranspose(*conv_arg, **conv_kwarg)
                            for conv_args, conv_kwargs in conv_params]
        self.dense_layers = [layers.Dense(*dense_args, **dense_kwargs)
                             for dense_args, dense_kwargs in dense_params]
