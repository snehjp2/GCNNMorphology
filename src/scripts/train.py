import os, sys
# I like to use typing, but you don't have to!
# from typing import Any, Dict, Tuple, Union
import argparse
import yaml
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import time

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import datasets, transforms
from e2cnn import gspaces
from e2cnn import nn as e2cnn_nn
from models import model_dict
from dataset import Galaxy10DECals
from tqdm import tqdm



def train_model(model, train_dataloader, test_dataloader, optimizer, epochs=100, device='cuda', save_dir='checkpoints', early_stopping_patience=10, report_interval=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    print("Model Loaded to Device!")
    best_test_acc = 0
    no_improvement_count = 0
    losses = []
    steps = []
    print("Training Started!")
    for epoch in range(epochs):
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader, 0), unit="batch", total=len(train_dataloader)):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            steps.append(epoch * len(train_dataloader) + i + 1)

        if (epoch + 1) % report_interval == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            test_acc = 100 * correct / total
            print(f"Epoch: {epoch + 1}, Accuracy: {test_acc:.2f}%")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                no_improvement_count = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch + 1}.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"))

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


def main(config):
    model = model_dict[config['model']]()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=config['parameters']['lr'], 
                            weight_decay=config['parameters']['weight_decay'])

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("Loading Dataset!")
    start = time.time()
    dataset = Galaxy10DECals(config['dataset'],transform)
    end = time.time()
    print(f"Dataset Loaded! in {end- start}")
    train_length=int(0.8* len(dataset))

    test_length=len(dataset)-train_length

    train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,test_length))
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['parameters']['batch_size'], shuffle=True)

    save_dir = config['save_dir'] + config['model']
    train_model(model, train_dataloader, test_dataloader, optimizer, epochs=config['parameters']['epochs'], device=device, save_dir=save_dir,early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 

    parser = argparse.ArgumentParser(description='Train the models')
    parser.add_argument('--config', metavar='config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
   
    main(config)
