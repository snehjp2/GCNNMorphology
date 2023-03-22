import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class Galaxy10DECals(Dataset):
    """Loading Galaxy10 DECals dataset from .h5 file.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str) :
        self.dataset_path = dataset_path
        self.dataset = None
        with h5py.File(self.dataset_path,"r") as f:
		self.length = len(f['ans'][()])
	
	


	
if __name__ == "__main__":
	print("Hey")	
