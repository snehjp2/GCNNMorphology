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

def load_models(directory_path):

    trained_models = dict.fromkeys(model_dict.keys())
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.pt') and os.path.isfile(file_path):
            model_name = os.path.splitext(file_name)[0]
            model = model_dict[str(model_name)]()
            model.load_state_dict(torch.load(file_path, map_location=device))
            model = nn.DataParallel(model)
            model = model.to(device)

            trained_models[model_name] = model
            
    trained_models = {key: value for key, value in trained_models.items() if value is not None}
 
    return trained_models

@torch.no_grad()
def compute_metrics(eval_loader: DataLoader, model: nn.Module, model_name: str, save_dir: str):
    
    classes = ('Disturbed Galaxies', 'Merging Galaxies', 
        'Round Smooth Galaxies', 'In-between Round Smooth Galaxies', 
        'Cigar Shaped Smooth Galaxies', 'Barred Spiral Galaxies', 
        'Unbarred Tight Spiral Galaxies', 'Unbarred Loose Spiral Galaxies', 
        'Edge-on Galaxies without Bulge', 'Edge-on Galaxies with Bulge')
    
    y_pred, y_true = [], []
    
    for batch in tqdm(eval_loader, unit="batch", total=len(eval_loader)):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        tmp = torch.tensor(labels == pred_labels, dtype=torch.float).mean()
        
        y_pred.extend(pred_labels)
        y_true.extend(labels.cpu().numpy())
    
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)    
    sklearn_report = classification_report(y_true, y_pred, output_dict=True, labels=classes)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"), bbox_inches='tight')
    
    return sklearn_report
    
@torch.no_grad()
def main():
    
    trained_models = load_models(args.path)
    print('Models Loaded!')
    
    model_metrics = dict.fromkeys(trained_models.keys())
    
    for model_name, model in tqdm(trained_models.items()):
        full_report = compute_metrics(eval_loader=test_dataloader, model=model, model_name=model_name, save_dir=args.path)
        
        model_metrics[model_name] = full_report

    print('Compiling All Metrics')
    with open(f'{args.path}/test_metrics.yaml', 'w') as file:
        yaml.dump(model_metrics, file)
    
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
    