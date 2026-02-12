# Unsupervised Representation Learning Generates Differentiable Neurophysiological Profiles

![Overview Figure](paper_figures/fig1%20-%20visual%20abstract.png)

Code for the paper ["Unsupervised Representation Learning Generates Differentiable Neurophysiological Profiles"](), (2026).

This repository contains all the scripts for training and post hoc experimentations presented in the paper.
To run the experiments, you must first have the datasets accessible and formatted in BIDS format. The dataset objects `aefp/datasets` can also be modified to fit different datasets.
Before running the training scripts, please verify the data and save paths.

Configuration files can be found under `conf`. [Weights & Biases](https://wandb.ai/site) is used to track the training progress and outputs. Set it up in the configuration files.

After configuring the model parameters in the configuration files, use `train/train.py` to train and save your first model. Use `train/fine_tune.py` to further fine-tuned the model if needed.

Model code can be found under `aefp/architecture`. Code for the experiments can be found under `experiments/scripts/`.


## Installation
Install dependencies via pip:
```shell
git clone https://github.com/mLapatrie/aefp.git
cd aefp
pip install -e .
```


