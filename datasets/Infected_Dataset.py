import os
from re import X
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
from torch.utils.data import Dataset

class Infected_Dataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        lens_data=None,
        n2n=False
        ):
        self.x, self.y = self._get_n2n_data(x_data, y_data) if n2n else (x_data, y_data)
        if type(lens_data) == type(None):
            self.lens = [len(i) for i in self.x]
        else:
            self.lens = lens_data
            
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        lens = self.lens[index]
        return x, y, lens
    
    def collate_fn(self, dataset):
        x, y, lens = zip(*dataset)
        if len(np.array(x[0]).shape) == 1:
            x_pad = torch.zeros(len(x), max(lens), 1).float()
        else: 
            x_pad = torch.zeros(len(x), max(lens), len(x[0][0])).float()
        
        for i, xx in enumerate(x):
            end = lens[i]
            x_pad[i,:end] = torch.FloatTensor(np.array(xx)).unsqueeze(1) if len(np.array(xx).shape) == 1 else torch.FloatTensor(np.array(xx))
        
        return x_pad, torch.FloatTensor(y), torch.LongTensor(lens)
    
    
    def _get_n2n_data(self, x, y):
        length = len(x)
        assert length == len(y)

        x_n2n = []
        y_n2n = []
        for i in range(length):
            for j in range(len(x[i])):
                x_n2n.append(np.array(x[i])[:j+1])
                y_n2n.append(y[i][j])
        return x_n2n, y_n2n
