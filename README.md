# Description

Here we try to recognise the letters in a captcha with an OCR model.
Firstly we wand to understand the structure of the project. Then we
create a DVC pipeline and experiment with it.


# Blogpost

The originial blogpost can be found [here](https://blog.jaysinha.me/train-your-first-neural-network-for-optical-character-recognition/).

# Data

The data can be downloaded [here](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images?resource=download).

# Original notebook

The original notebook was downloaded from [here](https://www.kaggle.com/code/razor08/ocr-for-captchas/notebook).


# The environment

can be found in the conda.txt file and installed with conda by ```conda create -naem <NAME> --file conda.txt```


# DVC

Adding pipeline steps to dvc works as follows

```
    dvc run --name <STAGENAME> \
    python <PATH_TO_SCRIPT> --config=params.yaml
    --deps <PATH_TO_INPUT>
    --outs <PATH_TO_OUTPUT>
    --params <FIELD in params.yaml>

```

The first stage e.g. is added as

```
dvc run --name split_data \
--deps data/raw \
--outs data/split/characterset.txt \
--outs data/split/x_train.txt \
--outs data/split/y_train.txt \
--outs data/split/x_valid.txt \
--outs data/split/y_valid.txt \
--params split \
--params base \
python src/stages/split.py --config params.yaml
```

Check out the two generate files! dvc.lock and dvc.yaml and note down what you find
in them.


Then add the other stages with the ```dvc run command```

```
dvc run --name create_datasets \
 --deps data/split/x_train.txt \
 --deps data/split/x_valid.txt \
 --deps data/split/y_train.txt \
 --deps data/split/y_valid.txt \
 --outs data/datasets/train_dataset \
 --outs data/datasets/validation_dataset \
 --params transform \
 --params base \
 python src/stages/datasets.py --config params.yaml
```

```
dvc run --name model_setup \
--deps data/split/characterset.txt \
--outs models/untrained_model.h5 \
--params model_setup \
--params base \
python src/stages/model_setup.py --config params.yaml
```

```
dvc run --name training \
 --deps models/untrained_model.h5 \
 --deps data/datasets/train_dataset \
 --deps data/datasets/validation_dataset \
 --outs models/trained_model.h5 \
 --outs models/prediction_model.h5 \
 --params train \
 --params base \
 python src/stages/training.py --config params.yaml
```