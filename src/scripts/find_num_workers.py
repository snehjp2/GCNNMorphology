import torch
from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader
from dataset import Galaxy10DECals

dataset = Galaxy10DECals("../../../Galaxy10_DECals.h5")

train_length=int(0.8* len(dataset))

test_length=len(dataset)-train_length

train_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,test_length))

for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))