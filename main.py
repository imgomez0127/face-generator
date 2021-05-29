#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vae import VAE

vae_kwargs = {
    "conv_amt": 5,
    "dense_amt": 10,
    "middle_dim": 512,
    "latent_dim": 256
}

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
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="Test image to check how the model compresses than decompresses"
    )
    parser.add_argument("--directory", type=str, help="Directory with training images")
    return parser.parse_args()

def get_iter(generator, kwargs):
    def f():
        return generator.flow_from_directory(**kwargs)
    return f
def train(directory, target_shape=(256, 256, 3), save_image=False):
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
    output_signature = (tf.TensorSpec(shape=(None, *target_shape), dtype=tf.float64))
    img_dataset = Dataset.from_generator(
        img_iter_gen,
        output_signature=output_signature
    )
    autoencoder = VAE(target_shape, regularization_weight=1e-3, **vae_kwargs)
    optimizer = RMSprop(learning_rate=1e-3)
    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        restore_best_weights=True
    )
    autoencoder.compile(optimizer=optimizer, run_eagerly=True)
    history = autoencoder.fit(
        x=img_dataset,
        steps_per_epoch=32,
        epochs=100,
        callbacks=[early_stop]
    )
    autoencoder.compile(optimizer=optimizer)
    autoencoder.save_weights("models/vae_model.h5")
    return autoencoder, history

def generate_image(target_shape=(256, 256, 3), save_image=False):
    autoencoder = VAE(target_shape, regularization_weight=1e-3, **vae_kwargs)
    autoencoder(tf.random.uniform([1, *target_shape]))
    autoencoder.compile(run_eagerly=True)
    autoencoder.load_weights("models/vae_model.h5")
    return autoencoder.generate_image()

def reconstruct_image(img, target_shape=(256, 256, 3)):
    autoencoder = VAE(target_shape, regularization_weight=1e-3, **vae_kwargs)
    autoencoder(tf.random.uniform([1, *target_shape]))
    autoencoder.compile(run_eagerly=True)
    autoencoder.load_weights("models/vae_model.h5")
    return autoencoder(tf.reshape(img, [1, *img.shape]))

if __name__ == "__main__":
    args = parse_args()
    if args.train:
        autoencoder, history = train(args.directory, save_image=args.save_image)
    elif args.test:
        image = np.asarray(Image.open(args.test)).astype(np.float32)
        reconstruction = reconstruct_image(image)
        plt.imshow(reconstruction[0])
        plt.show()
    else:
        image = generate_image().numpy()
        plt.imshow(image[0])
        plt.show()
