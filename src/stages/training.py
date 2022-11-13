from tensorflow import keras
import yaml
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from src.custom_layers.CTC import CTCLayer

def train(model, train_dataset, validation_dataset, log_dir='.', early_stopping = {}, fit = {}):

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(**early_stopping)

    # Tensorboard
    # tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(
        x = train_dataset,
        validation_data=validation_dataset,
        callbacks=[early_stopping],
        **fit
    )

    return history

def plot_history(history):

    fig,ax = plt.subplots(ncols=2)

    ax[0].plot(history.epoch[1:], history.history['loss'][1:])
    ax[0].plot(history.epoch[1:], history.history['val_loss'][1:])

    ax[1].semilogy(history.epoch[1:], history.history['loss'][1:])
    ax[1].semilogy(history.epoch[1:], history.history['val_loss'][1:])

    fig.tight_layout()
    return fig, ax

def train_from_config(config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file)

    image_width = config['base']['image_width']
    image_height = config['base']['image_height']

    train_dataset = tf.data.experimental.load(config['train']['inputs']['train_data'],
                                            {'image': tf.TensorSpec(shape=(None, image_width, image_height, 1), dtype=tf.float32, name=None),
                                             'label': tf.TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
                                            }
                                            )

    validation_dataset = tf.data.experimental.load(config['train']['inputs']['validation_data'],
                                            {'image': tf.TensorSpec(shape=(None, image_width, image_height, 1), dtype=tf.float32, name=None),
                                             'label': tf.TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
                                            }
                                            )

    model = keras.models.load_model(config['train']['inputs']['model'],
                                    custom_objects={'CTCLayer': CTCLayer})

    history = train(model = model,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    log_dir=config['train']['outputs']['tensorboard'],
                    early_stopping = config['train']['early_stopping'],
                    fit = config['train']['fit'])

    model.save(config['train']['outputs']['model'], save_format='h5')

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input,
        model.get_layer(name="dense2").output
    )

    prediction_model.save(config['train']['outputs']['prediction_model'], save_format='h5')

    fig, _ = plot_history(history)

    fig.savefig(config['train']['outputs']['history_plot'], dpi=400)


    return history, model

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()


    train_from_config(args.config_path)
    print("Finished predictions.")