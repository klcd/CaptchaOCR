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

