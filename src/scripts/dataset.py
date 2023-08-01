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
import copy



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
    def __init__(self,dataset_path : str, transform = None, extended=True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.extended = extended
        with h5py.File(self.dataset_path,"r") as f:
            self.img = f['images'][()]
            self.label = f['labels'][()]
            if extended:
                self.angle = f['angles'][()]
                self.redshift = f['redshifts'][()]
            self.length = len(f['labels'][()])
        if self.img.shape[1] == 3:
            self.img = np.transpose(self.img, (0,2,3,1))
        print(self.img.shape)
        print(self.label.shape)


    def __getitem__(self, idx):

        img = self.img[idx]
        label = torch.tensor(self.label[idx],dtype=torch.long)
        if self.extended:
            angle = torch.tensor(self.angle[idx],dtype=torch.long)
            redshift = torch.tensor(self.redshift[idx],dtype=torch.float)

        if self.transform:
            img = self.transform(img)

        if self.extended:
             return img, label, angle, redshift
        else:
            return img, label

    def __len__(self):
        return self.length

if __name__ == '__main__':
    
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
    '''
    train_dataset = Galaxy10DECals('/Users/snehpandya/Projects/GCNNMorphology/data/Galaxy10_DECals.h5')
    val_dataset = copy.deepcopy(train_dataset)
    
    indices = torch.randperm(len(train_dataset))
    val_size = int(len(train_dataset) * .2)
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
    assert len(train_dataset) + len(val_dataset) == len(indices)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    
    print(train_dataset.dataset.transform)
    print(val_dataset.dataset.transform)
    end_time = time.time()
    '''

    test_dataset = Galaxy10DECalsTest('/work/GDL/test_data_imbalanced.hdf5', val_transform)
    img, label, _, _ = test_dataset[0]

    print(img.shape)
    print(img.max())
    print(img.min())
