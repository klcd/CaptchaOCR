from tensorflow import keras
import yaml
import tensorflow as tf

def train(model, train_dataset, validation_dataset, log_dir='.', early_stopping = {}, fit = {}):

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(**early_stopping)

    # Tensorboard
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = model.fit(
        x = train_dataset,
        validation_data=validation_dataset,
        callbacks=[early_stopping, tb_callback],
        **fit
    )

    return history

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

    model = keras.models.load_model(config['train']['inputs']['model'])

    history = train(model = model,
                    train_dataset = train_dataset,
                    validation_dataset = validation_dataset,
                    log_dir=config['train']['outputs']['tensorboard'],
                    early_stopping = config['train']['early_stopping'],
                    fit = config['train']['fit'])

    model.save(config['train']['outputs']['model'], save_format='h5')



    return history, model