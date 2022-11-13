import numpy as np
import argparse
import yaml
from pathlib import Path
from typing import Text
import os
import pickle


def split_data(images, labels, train_size=0.9, shuffle=True, random_seed=42):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)

    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid

def split_data_from_config(config_path: Text):

    with open(config_path) as config_file:
        config = yaml.load(config_file)

    data_dir = Path(config['split']['input_dir'])
    images = np.array(sorted(list(map(str, list(data_dir.glob("*.png"))))))
    labels = np.array([img.split(os.path.sep)[-1].split(".png")[0] for img in images])

    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in labels])
    with open(config['split']['outputs']['max_length'], 'wb') as fid:
        pickle.dump(max_length,fid )

    # Save the unique characters
    characters = np.unique([char for label in labels for char in label])
    os.makedirs(config['split']['output_dir'], exist_ok=True)
    np.savetxt(config['split']['outputs']['characters'], characters, fmt='%s')

    x_train, x_valid, y_train, y_valid = split_data(images,
                                                    labels,
                                                    train_size=config['split']['train_fraction'],
                                                    shuffle=config['split']['shuffle'],
                                                    random_seed=config['base']['random_seed'])


    np.savetxt(config['split']['outputs']['x_train'], x_train, fmt='%s')
    np.savetxt(config['split']['outputs']['y_train'], y_train, fmt='%s')
    np.savetxt(config['split']['outputs']['x_valid'], x_valid, fmt='%s')
    np.savetxt(config['split']['outputs']['y_valid'], y_valid, fmt='%s')

    return x_train, x_valid, y_train, y_valid




if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config_path', required=True)
    args = arg_parser.parse_args()

    split_data_from_config(config_path=args.config_path)

    print("Finished splitting data.")