#!/usr/bin/env python3

import argparse
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vae import VAE

def parse_args():
    parser = argparse.ArgumentParser(description="File for running VAE")
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        description="Flag to set to training mode"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        description="Flag to save the model"
    )
    parser.add_argument("--directory", type=str, description="Directory with training images")
    return parser.parse_args()

def reconstruction_loss(x, y):
    return tfm.reduce_mean(tf.keras.binary_crossentropy(x, y))

def train(directory, target_shape=(256, 256, 3), save=True):
    save_dir = "./modified-faces" if save else None
    image_gen = ImageDataGenerator(
        featurewise_cetner=True,
        featurewise_std_normalization=True,
        horizontal_flip=True,

    )
    image_iter = image_gen.flow_from_directory(
        directory=directory,
        target_size=target_shape[:2],
        class_mode=None,
        seed=218,
        save_to_dir=save_dir,
        save_prefix="test",
        save_format="jpeg"

    )
    autoencoder = VAE(target_shape)
    optimizer = Adam(lr=0.001)
    autoencoder.compile(optimizer=optimizer, loss=reconstruction_loss)
    history = autoencoder.fit(x=image_iter, epochs=20, validation_split=0.2)
    if save:
        autoencoder.save("models/my_mode.h5")
    return autoencoder, history

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train(args.directory, save=args.save)
