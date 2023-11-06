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

*The techniques presented here have the potential to classify and extract features from images of arbitrary orientation and of significantly degraded quality, and as such warrant ethical concerns for maladaptations of this work. The exploitation of computer vision technologies for uses of surveillance is a poison. The authors steadfastly abhor the use of deep learning for purposes that do not seek to further scientific knowledge or provide a beneficial and equitable service to society.*

## Citation

```bibtex
@misc{pandya2023e2,
      title={E(2) Equivariant Neural Networks for Robust Galaxy Morphology Classification}, 
      author={Sneh Pandya and Purvik Patel and Franc O and Jonathan Blazek},
      year={2023},
      eprint={2311.01500},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA}
}
```
