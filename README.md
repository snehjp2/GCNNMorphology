# *E(2)* Equivariant Networks for Robust Galaxy Morphology Classification

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/snehjp2/GCNNMorphology/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## About

This repository includes code for constructing group convolutional neural networks (GCNNs) using [escnn](https://github.com/QUVA-Lab/escnn) for the purpose of robust galaxy morphology classification. Despite their remarkable abilities, deep neural networks have shown to be susceptible to indescernable amounts of noise or one-pixel perturbations, broadly known as *adversarial attacks*. We construct GCNNs for the task of galaxy morphology classification on the [Galaxy10DEcALS](https://github.com/henrysky/Galaxy10) dataset and study the performance of GCNNs as robust classifiers that are performant in the presence of strong noise and one-pixel perturbations which simulate limited observational capabilities and hardware defects common in astronomical imaging pipelines. The dataset features $17,736$ Galaxy images of 10 separate classes from observations from the [Sloan Digital Sky Survey](https://classic.sdss.org) and [DESI Legacy Imaging Surveys](https://www.legacysurvey.org). The dataset can be retrieved [here](https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5).

This project is developed for Python3.9 interpreter on a linux machine. Using an Anaconda virtual environment is recommended.

To install dependencies, run:

```console
conda env create -f environment.yml
```

or consult online documentation for appropriate dependencies.

## Statement on Broader Impact

*The algorithms presented here can extract features from and classify images of arbitrary orientation and suboptimal image quality. The authors condemn any maladaptations of this work and emphasize the importance of using AI technologies ethically and responsibly.*
