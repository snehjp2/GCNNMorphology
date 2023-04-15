import sys
import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T



class Galaxy10DECals(Dataset):
    """Loading Galaxy10 DECals dataset from .h5 file.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str, transform = None) :
        self.dataset_path = dataset_path
        #self.dataset = None
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.img = f['images'][()]
            self.label = f['labels'][()]
            self.length = len(f['labels'][()])

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
    def __init__(self,dataset_path : str, transform = None) :
        self.dataset_path = dataset_path
        self.dataset = None
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.length = len(f['labels'][()])

    def __getitem__(self, idx):

        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path,"r")

        img = self.dataset['images'][idx]
        label = torch.tensor(self.dataset['labels'][idx],dtype=torch.long)
        angle = torch.tensor(self.dataset['angles'][idx],dtype=torch.long)
        redshift = torch.tensor(self.dataset['redshifts'][idx],dtype=torch.float)
        
        if self.transform:
            img = self.transform(img)
        return img, label , angle, redshift

    def __len__(self):
        return self.length

if __name__ == '__main__':
    print(sys.argv[1])
    transform = T.ToTensor()
    dataset = Galaxy10DECals(sys.argv[1],transform=transform)
    #dataset = Galaxy10DECalsTest(sys.argv[1],transform=transform)
    print(len(dataset))
    img , label = dataset[1232]
    print(img.shape)
    print(label)