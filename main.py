#!/usr/bin/env python3

import argparse
import math
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.data import Dataset
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.vae import VAE
from models.gan import GAN

dataset_size = 600

vae_kwargs = {
    "conv_amt": 3,
    "dense_amt": 2,
    "middle_dim": 256,
    "latent_dim": 128,
    "regularization_weight": 1
}

gan_params = [
    8000, # latent_len
    (64, 64, 3), # Output Shape
    # Discriminator Args
    [
        [],
        {
            "conv_params":[
                ([8, 3], {"padding": "same", "strides": 2}),
                ([4, 3], {"padding": "same", "strides": 2}),
                ([3, 3], {"padding": "same", "strides": 2})
            ],
            "dense_params":
            [
                ([1], {})
            ]
        }
    ],
    # Generator Params
    [
        [256],
        {
            "conv_params": [
                ([128, 3], {"padding": "same"}),
                ([64, 3], {"padding": "same"}),
                ([32, 3], {"padding": "same"}),
                ([16, 3], {"padding": "same"}),
                ([3, 3], {"padding": "same"}),
            ]
        }
    ],
]

def parse_args():
    parser = argparse.ArgumentParser(description="File for running VAE")
    parser.add_argument(
        "--train",
        type=str,
        default="",
        help="Train image and provide directory with training images"
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
    parser.add_argument(
        "--image-count",
        type=int,
        default=1,
        help="Amount of images to generate"
    )
    return parser.parse_args()


def get_iter(generator, kwargs):
    def f():
        return generator.flow_from_directory(**kwargs)
    return f


def load_data(directory, target_shape=(64, 64, 3), save_image=False, batch_size=10, class_mode=None):
    save_dir = "./modified-faces" if save_image else None
    image_gen = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1.0/255,
        validation_split=0.2
    )
    iter_kwargs = {
        "directory": directory,
        "target_size": target_shape[:2],
        "class_mode": class_mode,
        "seed": 218,
        "save_to_dir": save_dir,
        "save_prefix": "test",
        "save_format": "jpeg",
        "subset": 'training',
        "batch_size": batch_size
    }
    img_iter_gen = get_iter(image_gen, iter_kwargs)
    output_signature = (tf.TensorSpec(shape=(None, *target_shape), dtype=tf.float32))
    img_dataset = Dataset.from_generator(
        img_iter_gen,
        output_signature=output_signature
    )
    return img_dataset


def train_vae(directory, target_shape=(64, 64, 3), save_image=False, batch_size=10):
    img_dataset = load_data(
        directory,
        target_shape=target_shape,
        save_image=save_image,
        batch_size=batch_size
    )
    autoencoder = VAE(target_shape, **vae_kwargs)
    optimizer = RMSprop(learning_rate=1e-3)
    early_stop = EarlyStopping(
        monitor='reconstruction_loss',
        min_delta=100,
        patience=10,
        restore_best_weights=True
    )
    autoencoder.compile(optimizer=optimizer, run_eagerly=True)
    history = autoencoder.fit(
        x=img_dataset,
        steps_per_epoch=math.ceil(dataset_size/batch_size),
        epochs=100,
        callbacks=[early_stop]
    )
    autoencoder.compile(optimizer=optimizer)
    autoencoder.save_weights("models/vae_model.h5")
    return autoencoder, history


def train_gan(directory, target_shape=(64, 64, 3), save_image=False, batch_size=30):
    img_dataset = load_data(
        directory,
        target_shape=target_shape,
        save_image=save_image,
        batch_size=batch_size,
    )
    gan = GAN(*gan_params)
    optimizer = Adam(learning_rate=1e-2)
    gan.compile(optimizer=optimizer, run_eagerly=True)
    history = gan.fit(
        x=img_dataset,
        steps_per_epoch=math.ceil(dataset_size/batch_size),
        epochs=300
    )
    gan.save_weights("models/gan_model.h5")
    return gan, history.history


def generate_image(target_shape=(64, 64, 3), save_image=False):
    autoencoder = VAE(target_shape, **vae_kwargs)
    autoencoder(tf.random.uniform([1, *target_shape]))
    autoencoder.compile(run_eagerly=True)
    autoencoder.load_weights("models/vae_model.h5")
    return autoencoder.generate_image()

def generate_image_gan(target_shape=(64, 64, 3)):
    gan = GAN(*gan_params)
    gan.discriminator((gan((tf.random.uniform((1, gan.latent_len)), None)), None))
    gan.compile(run_eagerly=True)
    gan.load_weights("models/gan_model.h5")
    return gan((tf.random.uniform((1, gan.latent_len)), None))


def reconstruct_image(img, target_shape=(64, 64, 3)):
    autoencoder = VAE(target_shape, **vae_kwargs)
    autoencoder(tf.random.uniform([1, *target_shape]))
    autoencoder.compile(run_eagerly=True)
    autoencoder.load_weights("models/vae_model.h5")
    return autoencoder(tf.reshape(img, [1, *img.shape]))


def get_largest_filenumber(files):
    largest_num = 1
    nums_regex = re.compile("[0-9]+")
    for f in files:
        match = nums_regex.match(f)
        if match:
                largest_num = max(int(match.group()), largest_num)
    return largest_num


def plot_losses(*args, file_dir="loss-curves"):
    for arg in args:
        plt.plot(range(len(arg)), arg)
    files = os.listdir(file_dir)
    plt.savefig(f'./{file_dir}/loss_curve{get_largest_filenumber(files)+1}.png')


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        autoencoder, history = train_gan(args.train, save_image=args.save_image, batch_size=100)
        plot_losses(history["gen_loss"], history["disc_loss"])
    elif args.test:
        image = np.asarray(Image.open(args.test)).astype(np.float32)
        reconstruction = reconstruct_image(image/255)
        plt.imshow(reconstruction[0])
        plt.savefig("reconstruction.png")
    else:
        for i in range(args.image_count):
            image = generate_image_gan().numpy()
            plt.imshow(image[0])
            plt.savefig(f"./generated-images/test{i}.png")
