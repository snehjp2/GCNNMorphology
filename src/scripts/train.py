import os, sys
# I like to use typing, but you don't have to!
# from typing import Any, Dict, Tuple, Union
import argparse
import yaml
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

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
from dataset import Galaxy10DECals, Galaxy10DECalsTest
from tqdm import tqdm
import random


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


def train_model(model, train_dataloader, test_dataloader, optimizer, scheduler = None, epochs=100, device='cuda', save_dir='checkpoints', early_stopping_patience=10, report_interval=5):
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
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader, 0), unit="batch", total=len(train_dataloader)):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses.append(loss.item())
            steps.append(epoch * len(train_dataloader) + i + 1)

        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            correct = 0
            total = 0
            test_loss = 0.0

            with torch.no_grad():
                for batch in test_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            test_acc = 100 * correct / total
            test_loss /= len(test_dataloader)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%, Learning rate: {lr}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                no_improvement_count = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pt"))
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
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches='tight')
    
    return best_test_acc, losses[-1]

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
def plot_confusion_matrix(data_loader: DataLoader, save_dir: str, model: nn.Module, device = 'cuda'):
    best_model = model
    best_model_path = f'{save_dir}/best_model.pt'
    
    best_model.load_state_dict(torch.load(best_model_path, map_location = device))
    y_pred = []
    y_true = []

    for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            output = best_model(inputs) # Feed Network
            pred_labels = torch.argmax(output, dim=-1).cpu().numpy()
            y_pred.extend(pred_labels) # Save Prediction
            y_true.extend(labels)
    
    classes = ('Disturbed Galaxies', 'Merging Galaxies', 
               'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
               'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
               'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
               'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), bbox_inches='tight')


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
    optimizer = optim.AdamW(params_to_optimize, lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'],gamma=config['parameters']['lr_decay'])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("Loading train dataset!")
    start = time.time()
    train_dataset = Galaxy10DECals(config['dataset']['train'],transform)
    end = time.time()
    print(f"dataset loaded in {end - start} s")
    train_length = int(0.8* len(train_dataset))

    val_length = len(train_dataset)-train_length

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,(train_length, val_length))
    train_dataloader = DataLoader(train_dataset, batch_size = config['parameters']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = config['parameters']['batch_size'], shuffle=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + config['model'] + '_' + timestr
    best_acc, final_loss = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=config['parameters']['epochs'], device=device, save_dir=save_dir,early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])

    # print("Loading test dataset!")
    # start = time.time()
    # test_dataset = Galaxy10DECalsTest(config['dataset']['test'], transform)
    # end = time.time()
    # print(f"Test dataset loaded in {end - start} s")
    # test_dataloader = DataLoader(test_dataset, batch_size = config['parameters']['batch_size'], shuffle=False)
    
    plot_confusion_matrix(data_loader = val_dataloader, save_dir = save_dir, model = model)
    
    config['best_acc'] = best_acc
    config['final_loss'] = final_loss
    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()
    
if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
        
    set_all_seeds(42)

    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
   
    main(config)
