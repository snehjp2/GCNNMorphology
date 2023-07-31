from test import load_models, compute_metrics
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
import h5py
from tqdm import tqdm

def compute_accuracy(test_loader: DataLoader, model: nn.Module, model_name: str, save_dir: str, output_name: str):
    
    y_pred, y_true = [], []
    
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, label, _, _ = batch
        input, label = input.to(device), label.to(device)
        outputs = model(input)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        
        y_pred.extend(pred_labels)
        y_true.extend(label.cpu().numpy())        
        indices = np.where(pred_labels == label.cpu().numpy())[0]
        
        return indices
        
def main(model_dir):
    
    trained_models = load_models(model_dir)
    print('All Models Loaded!')
    
    ## create dictionary with model names as keys and None as values
    indices_dict = {model_name: None for model_name in trained_models.keys()}
    
    for model_name, model in tqdm(trained_models.items()):
        indices_dict[model_name] = compute_accuracy(test_dataloader, model, model_name, args.model_dir, args.output_name)
        
    ## find the intersection of all the indices
    
    intersection = list(set.intersection(*map(set, indices_dict.values())))
    subset_images = torch.utils.data.Subset(test_dataset, intersection)
    subset_labels = [test_dataset[i][1] for i in intersection]
    return subset_images, subset_labels


if __name__ == '__main__':
    
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser(description = 'Path to Models and Data')
    parser.add_argument('--model_dir', metavar = 'model_dir', required=True,
                    help='Location of the model directory')
    
    parser.add_argument('--data_path', metavar = 'data_path', required=True, help='Location of the test data file')
    parser.add_argument('--output_name', metavar = 'output_name', required=True, help='Name of the output file')
    args = parser.parse_args()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(255)
        ])
    
    test_dataset = Galaxy10DECalsTest(str(args.data_path), transform)
    print("Test Dataset Loaded!")
    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)
    
    subset_images, subset_labels = main(args.model_dir)
    
    subset_images_np = torch.stack([img for img, _ in subset_images]).numpy()
    subset_labels_np = np.array(subset_labels)
    
    ## save subset as h5 file 
    
    def save_in_h5py(f, images, labels):
        dataset = f.create_dataset(
            "images", np.shape(images), data=images, compression='gzip', chunks=True)
        meta_set = f.create_dataset(
            "labels", np.shape(labels), data=labels,  compression='gzip', chunks=True)
        
    f = h5py.File(f'{args.model_dir}/{args.output_name}.h5','w')
    save_in_h5py(f, subset_images_np.astype(np.uint8), subset_labels_np)
    f.close()
    
    
