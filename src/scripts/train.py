import os
import argparse
import yaml
import numpy as np
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
from torchvision import transforms
from models import model_dict, feature_fields
from dataset import Galaxy10DECals
from tqdm import tqdm
import random


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler = None, epochs=100, device='cuda', save_dir='checkpoints', early_stopping_patience=10, report_interval=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = nn.DataParallel(model)
    model.to(device)
    
    print("Model Loaded to Device!")
    best_val_acc = 0
    no_improvement_count = 0
    losses = []
    steps = []
    print("Training Started!")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # for i, batch in tqdm(enumerate(train_dataloader, 0), unit="batch", total=len(train_dataloader)):
        for i, batch in tqdm(enumerate(train_dataloader)):
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
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            val_acc = 100 * correct / total
            val_loss /= len(val_dataloader)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Learning rate: {lr}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                torch.save(model.module.state_dict(), os.path.join(save_dir, f"best_model.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break

    torch.save(model.module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    
    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches='tight')
    
    return best_val_epoch, best_val_acc, losses[-1]

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
def plot_confusion_matrix(data_loader: DataLoader, save_dir: str, model: nn.Module):
    best_model = model
    best_model_path = f'{save_dir}/best_model.pt'
    
    best_model.load_state_dict(torch.load(best_model_path, map_location = device))
    best_model.to(device)
    y_pred = []
    y_true = []

    for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            output = best_model(inputs) # Feed Network
            pred_labels = torch.argmax(output, dim=-1).cpu().numpy()
            y_pred.extend(pred_labels) # Save Prediction
            y_true.extend(labels.cpu().numpy())
    
    classes = ('Disturbed Galaxies', 'Merging Galaxies', 
               'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
               'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
               'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
               'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')
    
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
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
        
def subsample(original_dataset, test_size):
    
    original_indices = np.asarray([x for x in range(17736)])
    augmented_indices = np.asarray([x for x in range(17736, len(original_dataset))])
    
    num_orig_train = int((1- test_size) * len(original_indices))
    num_orig_val = len(original_indices) - num_orig_train

    orig_train_indices = np.random.choice(original_indices, num_orig_train, replace=False)
    ind = np.zeros(len(original_indices), dtype=bool)
    ind[orig_train_indices] = True
    orig_val_indices = original_indices[~ind]
    
    train_inices = np.concatenate((orig_train_indices, augmented_indices))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_inices)
    val_sampler = torch.utils.data.SubsetRandomSampler(orig_val_indices)

    # Create data loaders for both the training and validation sets
    train_dataloader = DataLoader(original_dataset, batch_size=config['parameters']['batch_size'], sampler=train_sampler, pin_memory=True)
    val_dataloader = DataLoader(original_dataset, batch_size=config['parameters']['batch_size'], sampler=val_sampler, pin_memory=True)
    
    return train_dataloader, val_dataloader


def main(config):
    model = model_dict[config['model']]()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'],gamma=config['parameters']['lr_decay'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = Galaxy10DECals(config['dataset'], transform)
    end = time.time()
    print(f"dataset loaded in {end - start} s")
    
    train_dataloader, val_dataloader = subsample(train_dataset, 0.2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + config['model'] + '_' + timestr
    best_val_epoch, best_val_acc, final_loss = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=config['parameters']['epochs'], device=device, save_dir=save_dir,early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])
    print('Training Done')
    
    config['best_val_acc'] = best_val_acc
    config['best_val_epoch'] = best_val_epoch
    config['final_loss'] = final_loss
    config['feature_fields'] = feature_fields

    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()
    
    plot_confusion_matrix(data_loader = val_dataloader, save_dir = save_dir, model = model)

    
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
