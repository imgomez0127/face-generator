#!/usr/bin/env python3

import argparse
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.data import Dataset
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
        help="Flag to set to training mode"
    )
    parser.add_argument(
        "--save-image",
        type=bool,
        default=False,
        nargs="?",
        const=True,
        help="Flag to save the modified images"
    )
    parser.add_argument("--directory", type=str, help="Directory with training images")
    return parser.parse_args()

def get_iter(generator, kwargs):
    def f():
        return generator.flow_from_directory(**kwargs)
    return f
def train(directory, target_shape=(256, 256, 3), save_image=True):
    save_dir = "./modified-faces" if save_image else None
    image_gen = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1.0/255,
        validation_split=0.2
    )
    iter_kwargs = {
        "directory": directory,
        "target_size": target_shape[:2],
        "class_mode": None,
        "seed": 218,
        "save_to_dir": save_dir,
        "save_prefix": "test",
        "save_format": "jpeg",
        "subset": 'training'
    }
    img_iter_gen = get_iter(image_gen, iter_kwargs)
    img_dataset = Dataset.from_generator(
        img_iter_gen,
        output_signature=(tf.TensorSpec(shape=(None, *target_shape), dtype=tf.float64))
    )
    autoencoder = VAE(target_shape)
    optimizer = Adam(learning_rate=0.00001)
    autoencoder.compile(optimizer=optimizer, run_eagerly=True)
    history = autoencoder.fit(
        x=img_dataset,
        steps_per_epoch=32,
        epochs=20
    )
    autoencoder.save("models/my_model.h5")
    return autoencoder, history

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        autoencoder, history = train(args.directory, save_image=args.save_image)
