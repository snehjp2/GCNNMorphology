# E(2)-Equivariant Convolutional Neural Networks for Robust Galaxy Morphology Classifcation

*Contributors: Frank O, Sneh Pandya, Purvik Patel (all equal contribution)*

# Abstract

The increased use of supervised learning techniques in scientific analysis requires
architectures that can generalize well despite a potential data bottleneck, and which
can demonstrate robustness against diminishing quality of samples. In preparation
of the large influx of data from the upcoming Legacy Survey of Space and Time
(LSST) from the Vera Rubin Observatory, more advanced, robust, and effective
inference pipelines must be created to properly maximize LSST science. Here,
we propose the use of equivariant neural network architectures (GENNs), which
utilize the symmetries of the data as an inductive bias of the architecture itself, as a
candidate method in the problem of galaxy morphology classification.

This project is developed for Python3.9 interpreter on linux machine. Using an Anaconda virtual environment is recommended.

To install dependencies, simply run:

`pip install -r requirement.txt`

or consult online documentation for appropriate dependencies.

# Data

We train and validate our models on the open source [Galaxy10 DECals Dataset](https://github.com/henrysky/Galaxy10). The dataset features $17,736$ Galaxy images of 10 separate classes from observations from the Sloan Digital Sky Survey and DESI Legacy Imaging Surveys. The galaxies were classified through multiple rounds of volunteer voting with rigourous filtering through the [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) citizen science initiative.. Instructions on downloading the data can be found in the Galaxy10 DECals repository.

The classification of galaxy morphologies should be independent of the galaxy orientation in an example image. To test our baseline and equivariant models, we generate a rotated dataset by applying a random rotation parameterized by an angle &theta $\in (0, 2\pi)$ to the original dataset.
# Code

Our codebase utilizes the [e2cnn](https://github.com/QUVA-Lab/e2cnn) library, which is a Pytorch extension for equivariant deep learning.

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

To train a model, simply run:

`python train.py --config [path to config file]`

To test a model(s), simply run:

`python test.py --path [path to directory with trained moels]`

Any comments on this work are welcome. Please email [enter email].