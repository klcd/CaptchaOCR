import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from src.stages.encoding import encode_single_sample

@tf.autograph.experimental.do_not_convert
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

def get_decoding(characters):
    return layers.experimental.preprocessing.StringLookup(
                    vocabulary=list(characters),
                    mask_token=None, invert=True)


def plot_samples(train_dataset, num_to_char):

    fig, ax = plt.subplots(4, 4, figsize=(10, 5))
    for batch in train_dataset.take(1):
        images = batch["image"]
        labels = batch["label"]
        for i in range(16):
            img = (images[i] * 255).numpy().astype("uint8")
            label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
            ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    return fig, ax
def create_datasets_from_config(config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file)


    x_train = np.genfromtxt(config['transform']['inputs']['x_train'], dtype=str)
    y_train = np.genfromtxt(config['transform']['inputs']['y_train'], dtype=str)
    x_valid = np.genfromtxt(config['transform']['inputs']['x_valid'], dtype=str)
    y_valid = np.genfromtxt(config['transform']['inputs']['y_valid'], dtype=str)
    characters = list(np.genfromtxt(config['transform']['inputs']['characters'], dtype=str))

    train_dataset, validation_dataset = create_datasets(image_height = config['base']['image_height'],
                                                        image_width = config['base']['image_width'],
                                                        characters = characters,
                                                        batch_size = config['transform']['batch_size'],
                                                        x_train = x_train,
                                                        x_valid = x_valid,
                                                        y_train = y_train,
                                                        y_valid = y_valid)

    tf.data.experimental.save(train_dataset, config['transform']['outputs']['train_dataset'])
    tf.data.experimental.save(validation_dataset, config['transform']['outputs']['valid_dataset'])

    fig, _ = plot_samples(validation_dataset, num_to_char=get_decoding(characters))
    fig.savefig(config['transform']['outputs']['sample_plot'])


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()


    create_datasets_from_config(args.config_path)
    print("Finished datasets.")