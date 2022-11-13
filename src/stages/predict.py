import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import yaml
import pickle
import argparse

def get_decoding(characters):
    return layers.experimental.preprocessing.StringLookup(
                    vocabulary=list(characters),
                    mask_token=None, invert=True)

def decode_batch_predictions(pred, characters, max_length):
    '''
        A utility function to decode the output of the network
    '''

    num_to_char = get_decoding(characters)
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred,
                                       input_length=input_len,
                                       greedy=True)[0][0][:, :max_length]

    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def plot(batch_images, pred_texts, orig_texts ):

    acc_score = 0

    fig, ax = plt.subplots(4, 4, figsize=(15, 5))

    for i, text in enumerate(pred_texts):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        if str(pred_texts[i]) == orig_texts[i]:
            acc_score+=1
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
    return fig, ax


def batch_predict(prediction_model, characters, max_length, validation_dataset):

    # Mapping integers back to original characters
    num_to_char = get_decoding(characters)
    #Let's check results on some validation samples
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds, characters, max_length)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label)

    return batch_images, orig_texts, pred_texts

def predict_from_config(config_path):

    with open(config_path) as config_file:
        config = yaml.load(config_file)

    prediction_model = keras.models.load_model(config['predict']['inputs']['prediction_model']
                                    )




    image_width = config['base']['image_width']
    image_height = config['base']['image_height']

    validation_dataset = tf.data.experimental.load(config['predict']['inputs']['valid_dataset'],
                                            {'image': tf.TensorSpec(shape=(None, image_width, image_height, 1), dtype=tf.float32, name=None),
                                             'label': tf.TensorSpec(shape=(None, None), dtype=tf.int64, name=None)
                                            }
                                            )
    characters = np.genfromtxt(config['predict']['inputs']['characters'], dtype=str)

    #with open(config['predict']['inputs']['max_length'],'rb') as fid:
    #max_length = pickle.load(config['predict']['inputs']['max_length'])
    max_length = 5

    batch_images, orig_texts, pred_texts = batch_predict(prediction_model,
                                                         characters,
                                                         max_length,
                                                         validation_dataset)

    fig, ax = plot(batch_images, pred_texts, orig_texts )

    print(config['predict']['outputs']['sample_predictions'])
    fig.savefig(config['predict']['outputs']['sample_predictions'], dpi=400)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()


    predict_from_config(args.config_path)
    print("Finished predictions.")