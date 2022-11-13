from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import yaml
import argparse
import numpy as np
import os

from src.custom_layers.CTC import CTCLayer


def create_model(image_width, image_height, characters, optimizer, model_name = 'ocr_model', layer_params = {} ):

    input_img = layers.Input(shape=(image_width, image_height, 1),
                             name="image", dtype="float32")

    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(**layer_params['conv1'])(input_img)

    x = layers.MaxPooling2D(**layer_params['max_pooling1'])(x)

    # Second conv block
    x = layers.Conv2D(**layer_params['conv2'])(x)
    x = layers.MaxPooling2D(**layer_params['max_pooling2'])(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((image_width // 4), (image_height // 4) * 64)

    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(**layer_params['dense1'])(x)
    x = layers.Dropout(**layer_params['dropout1'])(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(**layer_params['bidirec_lstm1']))(x)
    x = layers.Bidirectional(layers.LSTM(**layer_params['bidirec_lstm2']))(x)

    # Output layer
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name=model_name
    )
    # Optimizer
    opt = eval(optimizer)
    # Compile the model and return
    model.compile(optimizer=opt)

    return model


def create_model_from_config(config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file)

    characters = np.genfromtxt(os.path.join(config['model_setup']['input'], 'characterset.txt'), dtype=str)

    model = create_model(image_height=config['model_setup']['image_height'],
                         image_width=config['model_setup']['image_width'],
                         optimizer=config['model_setup']['optimizer'],
                         characters=characters,
                         model_name=config['model_setup']['model_name'],
                         layer_params = config['model_setup']['layer_params'])

    model.save(os.path.join(config['model_setup']['output'], 'untrained_model.h5'),
               save_format='h5')

    return model

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()

    create_model_from_config(args.config_path)
    print("Finished model setup.")
