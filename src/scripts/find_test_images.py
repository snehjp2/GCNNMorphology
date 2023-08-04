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


def correct_classified_indices(test_dataset: Galaxy10DECalsTest, model: nn.Module, device: str = 'cuda'):

    test_dataloader = DataLoader(test_dataset, batch_size = 128, shuffle=False)

    y_pred, y_true = [], []
    
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    
    for batch in tqdm(test_dataloader, unit="batch", total=len(test_dataloader)):
        input, label, _, _ = batch
        input, label = input.to(device), label.to(device)
        outputs = model(input)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()

        y_pred.extend(pred_labels)
        y_true.extend(label.cpu().numpy())    

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)    
    indices = np.where(y_pred == y_true)[0]
        
    return indices
        
def main(model_dir):
    
    trained_models = load_models(model_dir)
    print('All Models Loaded!')
    
    ## create dictionary with model names as keys and None as values
    indices_dict = {model_name: None for model_name in trained_models.keys()}
    
    for model_name, model in tqdm(trained_models.items()):
        indices_dict[model_name] = correct_classified_indices(test_dataset, model)
        
    ## find the intersection of all the indices
    
    intersection = list(set.intersection(*map(set, indices_dict.values())))
    intersection_set = set(intersection)  # create a set for faster lookups

    subset_images = []
    subset_labels = []

    for idx in range(len(test_dataset)):
        if idx in intersection_set:
            image, label, _, _ = test_dataset[idx]
            subset_images.append(image.cpu().numpy())
            subset_labels.append(label.cpu().numpy())
    
    print(f"Number of images in the subset: {len(subset_images)}")
    print(f"Number of images in original test set: {len(test_dataset)}")
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
    
    subset_images, subset_labels = main(args.model_dir)
    
    subset_images_np = np.array(subset_images)
    subset_labels_np = np.array(subset_labels)
    
    print(len(subset_images_np), len(subset_labels_np))
    
    ## save subset as h5 file 
    
    def save_in_h5py(f, images, labels):
        dataset = f.create_dataset(
            "images", np.shape(images), data=images, compression='gzip', chunks=True)
        meta_set = f.create_dataset(
            "labels", np.shape(labels), data=labels,  compression='gzip', chunks=True)
        
    f = h5py.File(f'{args.model_dir}/{args.output_name}.h5','w')
    save_in_h5py(f, subset_images_np.astype(np.uint8), subset_labels_np)
    f.close()
    
    
