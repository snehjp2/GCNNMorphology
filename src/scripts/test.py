from models import model_dict
import argparse
import torch
import os
from dataset import Galaxy10DECalsTest
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torchvision import transforms
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from tqdm import tqdm

"""
input: path to directory containing trained models in .pt format. 
       Model file names must match dictionary keys from models.py.

returns: .yaml file with accuracy, precision, recall, and f1-score.
         Confusion matrix for each model on test set.
"""

def load_models(directory_path):

    trained_models = dict.fromkeys(model_dict.keys())
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.pt') and os.path.isfile(file_path):
            model_name = os.path.splitext(file_name)[0]
            model = model_dict[str(model_name)]()
            model.load_state_dict(torch.load(file_path, map_location = device))

            trained_models[model_name] = model
            
    trained_models = {key: value for key, value in trained_models.items() if value is not None}
 
    return trained_models

@torch.no_grad()
def compute_metrics(eval_loader: DataLoader, model: nn.Module):
    
    accuracy = []
    y_pred, y_true = [], []
    model = model.to(device)
    
    for batch in eval_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        tmp = torch.Tensor(labels == pred_labels, dtype=torch.float).mean()
        accuracy.append(tmp.item())
        
        y_pred.extend(pred_labels)
        y_true.extend(labels.cpu().numpy())
        
    accuracy = np.mean(accuracy)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    
    return accuracy, precision, recall, f1

# @torch.no_grad()
# def compute_accuracy(eval_loader: DataLoader, model: nn.Module):

#     accuracy = []

#     for batch in tqdm(eval_loader):
#         inputs, labels = batch[0].to(device), batch[1].to(device)
#         outputs = model(inputs)
#         pred_labels = torch.argmax(outputs, dim=-1)
#         tmp = (labels == pred_labels).float().mean()
#         accuracy.append(tmp.item())

#     accuracy = np.mean(accuracy)
#     return accuracy

# @torch.no_grad()
# def compute_precision(eval_loader: DataLoader, model: nn.Module):
    
#     y_pred, y_true = [], []
    
#     for batch in eval_loader:
#         inputs, labels = batch[0].to(device), batch[1].to(device)
#         outputs = model(inputs)
#         pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        
#         y_pred.extend(pred_labels)
#         y_true.extend(labels.cpu().numpy())
        
#     precision = precision_score(y_true, y_pred, average='binary')
#     return precision
    
# @torch.no_grad()    
# def compute_recall(eval_loader: DataLoader, model: nn.Module):
    
#     y_pred, y_true = [], []
    
#     for batch in eval_loader:
#         inputs, labels = batch[0].to(device), batch[1].to(device)
#         outputs = model(inputs)
#         pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        
#         y_pred.extend(pred_labels)
#         y_true.extend(labels.cpu().numpy())
        
#     recall = recall_score(y_true, y_pred, average='binary')
#     return recall

# @torch.no_grad()
# def compute_f1_score(eval_loader: DataLoader, model: nn.Module):
    
#     y_pred, y_true = [], []
    
#     for batch in eval_loader:
#         inputs, labels = batch[0].to(device), batch[1].to(device)
#         outputs = model(inputs)
#         pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        
#         y_pred.extend(pred_labels)
#         y_true.extend(labels.cpu().numpy())
        
#     f1 = f1_score(y_true, y_pred, average='binary')
#     return f1

@torch.no_grad()
def plot_confusion_matrix(data_loader: DataLoader, save_dir: str, model_dict: dict):
    
    classes = ('Disturbed Galaxies', 'Merging Galaxies', 
            'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
            'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
            'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
            'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')

    for model_name, model in model_dict.items():
        
        y_pred = []
        y_true = []
        
        for batch in data_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            output = model(inputs) # Feed Network
            pred_labels = torch.argmax(output, dim=-1).cpu().numpy()
            y_pred.extend(pred_labels) # Save Prediction
            y_true.extend(labels.cpu().numpy())
    
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.title(f'{model_name} Confusion Matrix')
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"), bbox_inches='tight')

    
@torch.no_grad()
def main():
    
    trained_models = load_models(args.path)
    print('Models Loaded!')
    
    model_accs = dict.fromkeys(trained_models.keys())
    model_precision = dict.fromkeys(trained_models.keys())
    model_recall = dict.fromkeys(trained_models.keys())
    model_f1_score = dict.fromkeys(trained_models.keys())
    
    for model_name, model in tqdm(trained_models.items()):
        accuracy, precision, recall, f1 = compute_metrics(eval_loader=test_dataloader, model=model)
        
        model_accs[model_name] = accuracy
        model_precision[model_name] = precision
        model_recall[model_name] = recall
        model_f1_score[model_name] = f1
        
    all_metrics = {'model_accuracy': model_accs, 'model_precision': model_precision, 
                    'model_recall': model_recall, 'model_f1_score': model_f1_score}
        
    plot_confusion_matrix(data_loader=test_dataloader, save_dir=args.path, model=model)
        
    print('Compiling All Metrics')
    with open(f'{args.path}/test_metrics.yaml', 'w') as file:
        yaml.dump(all_metrics, file)
    

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
    
    parser = argparse.ArgumentParser(description = 'Path to Models')
    parser.add_argument('--path', metavar = 'path', required=True,
                    help='Location of the model directory')

    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    test_path = '/Users/snehpandya/Projects/GCNNMorphology/data/random_rotations.hdf5'
    test_dataset = Galaxy10DECalsTest(test_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)
    
    main()
    