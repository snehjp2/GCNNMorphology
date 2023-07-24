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
import copy

def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)

def train_model(model, train_dataloader, val_dataloader, optimizer, model_name, scheduler = None, epochs=100, device='cuda', save_dir='checkpoints', early_stopping_patience=10, report_interval=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = nn.DataParallel(model)
    model.to(device)
    
    print("Model Loaded to Device!")
    best_val_acc, no_improvement_count = 0, 0
    losses, steps = [], []
    print("Training Started!")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
            correct, total, val_loss = 0, 0, 0.0

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
                torch.save(model.eval().module.state_dict(), os.path.join(save_dir, f"best_model.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break

    torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    np.save(os.path.join(save_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(save_dir, f"steps-{model_name}.npy"), np.array(steps))

    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches='tight')
    
    return best_val_epoch, best_val_acc, losses[-1]

def train_model_da(model, train_dataloader, val_dataloader, optimizer, model_name, scheduler = None, epochs=100, device='cuda', alpha=0.5, report_interval=5, save_dir='checkpoints', early_stopping_patience=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = nn.DataParallel(model)
    model.to(device)

    print("Model Loaded to Device!")
    best_val_acc, no_improvement_count = 0, 0
    losses, steps = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total = 0
        correct = 0

        for (source_inputs, source_targets), (target_inputs, _) in zip(train_dataloader, val_dataloader):
            source_inputs, source_targets = source_inputs.to(device), source_targets.to(device)
            target_inputs = target_inputs.to(device)

            optimizer.zero_grad()
            source_outputs = model(source_inputs)
            target_outputs = model(target_inputs)

            classification_loss = F.cross_entropy(source_outputs, source_targets)
            domain_adaptation_loss = torch.abs(source_outputs.mean() - target_outputs.mean())
            
            total_loss = classification_loss + alpha * domain_adaptation_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            losses.append(loss.item())
            
        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
        
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            correct, total, val_loss = 0, 0, 0.0

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
            print(f"Epoch: {epoch + 1}, Validation Loss (No DA): {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                torch.save(model.eval().module.state_dict(), os.path.join(save_dir, f"best_model.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break

    torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    np.save(os.path.join(save_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(save_dir, f"steps-{model_name}.npy"), np.array(steps))
    
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
        

def main(config):
    model_name = config['model']
    model = model_dict[config['model']]()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'],gamma=config['parameters']['lr_decay'])
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize(255)
    ])
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = Galaxy10DECals(config['dataset'])
    val_dataset = copy.deepcopy(train_dataset)

    indices = torch.randperm(len(train_dataset))
    val_size = int(len(train_dataset) * config['parameters']['test_size'])
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    assert len(train_dataset) + len(val_dataset) == len(indices)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    end = time.time()
    print(f"dataset loaded in {end - start} s")
    
    test_len = int(config['parameters']['test_size'] * len(train_dataset))
    train_len = len(train_dataset) - test_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, test_len])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + config['model'] + '_' + timestr
    best_val_epoch, best_val_acc, final_loss = train_model(model, train_dataloader, val_dataloader, optimizer, model_name, scheduler, epochs=config['parameters']['epochs'], device=device, save_dir=save_dir,early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])
    print('Training Done')
    
    config['best_val_acc'] = best_val_acc
    config['best_val_epoch'] = best_val_epoch
    config['final_loss'] = final_loss
    config['feature_fields'] = feature_fields

    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()
    
    plot_confusion_matrix(data_loader = val_dataloader, save_dir = save_dir, model = model)


def main_da(config):
    model_name = config['model']
    model = model_dict[config['model']]()
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = config['parameters']['milestones'],gamma=config['parameters']['lr_decay'])
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize(255)
    ])
    
    print("Loading train dataset!")
    start = time.time()
    train_dataset = Galaxy10DECals(config['dataset'])
    val_dataset = copy.deepcopy(train_dataset)

    indices = torch.randperm(len(train_dataset))
    val_size = int(len(train_dataset) * config['parameters']['test_size'])
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    assert len(train_dataset) + len(val_dataset) == len(indices)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    end = time.time()
    print(f"dataset loaded in {end - start} s")
    
    test_len = int(config['parameters']['test_size'] * len(train_dataset))
    train_len = len(train_dataset) - test_len
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_len, test_len])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dir = config['save_dir'] + config['model'] + '_' + timestr
    best_val_epoch, best_val_acc, final_loss = train_model_da(model, train_dataloader, val_dataloader, optimizer, model_name, scheduler, epochs=config['parameters']['epochs'], device=device, alpha=0.5, save_dir=save_dir, early_stopping_patience=config['parameters']['early_stopping'], report_interval=config['parameters']['report_interval'])
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

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_all_seeds(42)

    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main_da(config)
