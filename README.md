# E(2)-Equivariant Steerable Convolutional Neural Networks for Robust Galaxy Morphology Classification

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/snehjp2/GCNNMorphology/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Contributors: Frank O, [Sneh Pandya](https://snehjp2.github.io), Purvik Patel (all equal contribution)*

# Abstract

The increased use of supervised learning techniques in astronomical data analysis requires
architectures that can generalize well despite a potential data bottleneck, and which
can demonstrate robustness against diminishing quality of data. In preparation
of the large influx of data from the upcoming Legacy Survey of Space and Time
(LSST) from the [Vera Rubin Observatory](https://www.lsst.org), more advanced, robust, and effective
inference pipelines must be created to properly maximize LSST science. Here,
we propose the use of group convolutional neural network architectures (GCNNs), which
utilize the symmetries of the data as an inductive bias of the architecture itself, as a
candidate method in the problem of galaxy morphology classification.

This project is developed for Python3.9 interpreter on a linux machine. Using an Anaconda virtual environment is recommended.

To install dependencies, simply run:

`conda env create -f environment.yml`

or consult online documentation for appropriate dependencies.

# Data

We train and validate our models on the open source [Galaxy10 DECals Dataset](https://github.com/henrysky/Galaxy10). The dataset features $17,736$ Galaxy images of 10 separate classes from observations from the [Sloan Digital Sky Survey](https://classic.sdss.org) and [DESI Legacy Imaging Surveys](https://www.legacysurvey.org). The galaxies were classified through multiple rounds of volunteer voting with rigourous filtering through the [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) citizen science initiative. Instructions on downloading the data can be found in the Galaxy10 DECals repository.

The classification of galaxy morphologies should be independent of the galaxy orientation in an image. To test the classifcation accuracy of our baseline and equivariant models, we generate a rotated dataset by applying a random rotation parameterized by an angle $\theta \in (0, 2\pi]$ to images from the original dataset.
# Code

Our codebase utilizes the [e2cnn](https://github.com/QUVA-Lab/e2cnn) library, which is a [PyTorch](https://pytorch.org) extension for equivariant deep learning.

`src/notebooks/loading_data.ipynb`
Exploratory data analysis, morphological opening, and noise generation.

`src/scripts/e2wrn.py`
Equivariant Wide ResNet models as defined in the e2cnn library. We thank the author's of the repository for open-sourcing this code.

`src/scripts/e2resnet.py`
Equivariant ResNet model generously provided by David Klee. We thank David for sharing this model.

`src/scripts/models.py`
Defines C2, C4, C8, C16, D2, D4, D8, and D16-equivariant architectures. Pretrained ResNet18, ResNet50, and Wide ResNet from PyTorch.

`src/scripts/dataset.py`, `src/scripts/train.py`, `src/scripts/test.py`
Code to train, validate, and test models as defined in .yaml files in `src/config/`

To train a model, run:

`python train.py --config [path to model.yaml file]`

To test a model(s), run:

`python test.py --path [path to directory with trained models]`

Any comments on this work are welcome. Please email pandya.sne AT northeastern DOT edu.