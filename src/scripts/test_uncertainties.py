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
from sklearn.metrics import confusion_matrix, classification_report
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
classes = ('Disturbed Galaxies', 'Merging Galaxies', 
        'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
        'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
        'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
        'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')
    
    
def load_models(directory_path):

    trained_models = dict.fromkeys(model_dict.keys())
    device = ('cuda' if torch.cuda.is_available() else 'cpu') 
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.pt') and os.path.isfile(file_path):
            
            print(f'Loading {file_name}...')
            model_name = os.path.splitext(file_name)[0]
            model = model_dict[str(model_name)]()
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))
            print(f'Finishing Loading {model_name}')

            trained_models[model_name] = model
            
    trained_models = {key: value for key, value in trained_models.items() if value is not None}
 
    return trained_models

@torch.no_grad()
def compute_metrics(test_loader: DataLoader, model: nn.Module, model_name: str, save_dir: str, output_name: str):
    
    y_pred, y_true = [], []
    
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    accs = []
    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, label, _, _ = batch
        input, label = input.to(device), label.to(device)
        outputs = model(input)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        
        val_acc = 100 * correct / total
        accs.append(val_acc)
        
    mean_accs = float(np.mean(accs))
    std_accs = float(np.std(accs) / np.sqrt(len(accs)))
    
    averaged_report = {'mean_acc': mean_accs, 'std_acc': std_accs}
    
    return averaged_report
    
@torch.no_grad()
def main(model_dir):

        trained_models = load_models(model_dir)
        print('All Models Loaded!')
        
        model_metrics = dict.fromkeys(trained_models.keys())
        
        for model_name, model in tqdm(trained_models.items()):
            test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=True)
            full_report = compute_metrics(test_loader=test_dataloader, model=model, model_name=model_name, save_dir=model_dir)
            
            model_metrics[model_name] = full_report

        print('Compiling All Metrics')
        with open(f'{model_dir}/results_std.yaml', 'w') as file:
            yaml.dump(model_metrics, file)
    
if __name__ == '__main__':
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description = 'Path to Models and Data')
    
    parser.add_argument('--model_path', metavar = 'model_path', required=True,
                    help='Location of the model directory')
    
    parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')
    
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize(255)
    ])
    
    test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
    print("Test Dataset Loaded!")
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=True)
    
    main(args.model_path)
    