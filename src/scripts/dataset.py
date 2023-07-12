import sys
import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
from torchvision import transforms



class Galaxy10DECals(Dataset):
    """Loading Galaxy10 DECals dataset from .h5 file.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str, transform = None) :
        self.dataset_path = dataset_path
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.img = f['images'][()]
            self.label = f['ans'][()]
            self.length = len(f['ans'][()])

    def __getitem__(self, idx):

        img = self.img[idx]
        label = torch.tensor(self.label[idx],dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.length


class Galaxy10DECalsTest(Dataset):
    """Loading Galaxy10 DECals test dataset from .h5 file.
    Test dataset has original images roated at random angles.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str, transform = None):
        self.dataset_path = dataset_path
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.img = f['images'][()]
            self.label = f['labels'][()]
            self.angle = f['angles'][()]
            self.redshift = f['redshifts'][()]
            self.length = len(f['labels'][()])
            

    def __getitem__(self, idx):

        img = self.img[idx]
        label = torch.tensor(self.label[idx],dtype=torch.long)
        angle = torch.tensor(self.angle[idx],dtype=torch.long)
        redshift = torch.tensor(self.redshift[idx],dtype=torch.float)
        
        if self.transform:
            img = self.transform(img)
        return img, label, angle, redshift

    def __len__(self):
        return self.length

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(255),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    start_time = time.time()
    dataset = Galaxy10DECalsTest(sys.argv[1],transform=transform)
    end_time = time.time()
    print("Time taken to load dataset: ", end_time-start_time)
    
    start_time = time.time()
    for i in range(len(dataset)):
        img ,label, _, _ = dataset[i]
    end_time = time.time()
    print("Time taken to load all images: ", end_time-start_time)