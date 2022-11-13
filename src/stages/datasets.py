import argparse
import yaml
import numpy as np
import os
import json

import tensorflow as tf
from tensorflow.keras import layers
from src.stages.encoding import encode_single_sample

def create_datasets(image_height, image_width, characters, batch_size,
                     x_train, x_valid, y_train, y_valid):

    char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters),
                                                                 num_oov_indices=0,
                                                                 mask_token=None
        )

    encoding = lambda img_path, label: encode_single_sample(img_path,
                                                            label,
                                                            image_height,
                                                            image_width,
                                                            char_to_num)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encoding,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encoding,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return train_dataset, validation_dataset


def create_datasets_from_config(config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file)


    x_train = np.genfromtxt(os.path.join(config['transform']['input'], 'x_train.txt'), dtype=str)
    y_train = np.genfromtxt(os.path.join(config['transform']['input'], 'y_train.txt'), dtype=str)
    x_valid = np.genfromtxt(os.path.join(config['transform']['input'], 'x_valid.txt'), dtype=str)
    y_valid = np.genfromtxt(os.path.join(config['transform']['input'], 'y_valid.txt'), dtype=str)
    characters = np.genfromtxt(os.path.join(config['transform']['input'], 'characterset.txt'), dtype=str)

    train_dataset, validation_dataset = create_datasets(image_height = config['base']['image_height'],
                                                        image_width = config['base']['image_width'],
                                                        characters = characters,
                                                        batch_size = config['transform']['batch_size'],
                                                        x_train = x_train,
                                                        x_valid = x_valid,
                                                        y_train = y_train,
                                                        y_valid = y_valid)

    print(train_dataset.element_spec)

    tf.data.experimental.save(train_dataset, os.path.join(config['transform']['output'], 'train_dataset'))
    tf.data.experimental.save(validation_dataset, os.path.join(config['transform']['output'], 'validation_dataset'))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()


    create_datasets_from_config(args.config_path)
    print("Finished datasets.")