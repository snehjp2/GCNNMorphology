import os, sys
# I like to use typing, but you don't have to!
# from typing import Any, Dict, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import shift
import scipy.sparse as sp
# import tensorflow as tf
# import tensorflow_datasets as tfds
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import datasets, transforms
from e2cnn import gspaces
from e2cnn import nn as e2cnn_nn

# We might use tensorflow to load datasets.
# Prevent tensorflow from stealing our GPU.
# tf.config.experimental.set_visible_devices([], "GPU")

# Interactive plots.
# %matplotlib inline




def train(train_loader: DataLoader, model: nn.Module,
          optimizer: optim.Optimizer, epochs: int):


    losses = []
    # 1 epoch = 1 pass through the dataset.
    for epoch in range(epochs):
        print("Epoch {:d} / {:d}".format(epoch, epochs))
        # train_loader steps once it iterates over all data.
        for train_step, batch in enumerate(train_loader):
            # train_loader gives us batched images (CxHxW) and labels (integers).
            # If we are using a GPU we need to move them to the device.
            inputs, labels = batch[0].to(device), batch[1].to(device)
            # Reset last optimization step.
            optimizer.zero_grad()
            # Make prediction, calculate loss.
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            # Compute gradients and update weights.
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if train_step > 0 and train_step % 50 == 0:
                print("Mean of last 50 losses: {:f}".format(np.mean(losses[-50:])))

    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.savefig('../../plots/train_loss_saved.png')

# We don't need gradients during evaluation.
@torch.no_grad()
def evaluate(eval_loader: DataLoader, model: nn.Module):

    accuracy = []

    for batch in eval_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        pred_labels = torch.argmax(outputs, dim=-1)
        tmp = (labels == pred_labels).float().mean()
        accuracy.append(tmp.item())

    # We compute the mean of means over batches.
    # This could be slightly skewed if the last batch is smaller.
    # Does not matter too much here.
    accuracy = np.mean(accuracy)
    print("Correct answer in {:.1f}% of cases.".format(accuracy * 100))




@torch.no_grad()
def plot_predictions(eval_loader: DataLoader, model: nn.Module):
    example = next(iter(eval_loader))
    inputs = example[0].to(device)
    outputs = model(inputs)
    pred_labels = torch.argmax(outputs, dim=-1).to("cpu").numpy()

    plt.figure(figsize=(10, 10))
    for i in range(5*5):
        plt.subplot(5, 5, 1 + i)
        plt.title("Label: {:d}".format(pred_labels[i]))
        plt.imshow(example[0][i][0])
        plt.axis("off")
        plt.savefig('../../plots/eval_saved.png')


  
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the models')
    parser.add_argument('--train_loader', metavar='train_loader', required=True,
                    help='Train Loader')
    parser.add_argument('--model', metavar='model', required=True,
                    help='Model')
    parser.add_argument('--optimizer', metavar='optimizer', required=True,
                        help='Optimizer')
    parser.add_argument('--epochs', type=int, metavar='epochs', required=True,
                        help='Epochs')
    args = parser.parse_args()

    train(train_loader= args.train_loader, model= args.model,
          optimizer= args.optimizer, epochs= args.epochs)