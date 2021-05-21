#!/usr/bin/env python3
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2d, Conv2DTranspose, Flatten, Dense, Layer

class Encoder(Layer):

    def __init__(self, conv_amt=3, middle_dim=256, latent_dim=128):
        super().__init__()
        self.convs = [Conv2d(3, 3, padding='same') for _ in range(conv_amt)]
        self.dense = Dense(middle_dim)
        self.latent = Dense(latent_dim)

class Decoder(Layer):

    def __init__(self):
        super().__init__()

class VAE(Model):

    def __init__(self):
        super().__init__()
