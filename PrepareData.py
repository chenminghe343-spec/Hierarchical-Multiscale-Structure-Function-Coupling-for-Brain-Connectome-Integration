import torch
import os
import numpy as np


class MulSparseDataset(torch.utils.data.Dataset):
    def __init__(self, sc_path: str, fc_path: str, label_path: str, task='reg'):
        self.sc = torch.from_numpy(np.load(sc_path).astype(np.float32))
        self.fc = torch.from_numpy(np.load(fc_path).astype(np.float32))
        y = torch.from_numpy(np.load(label_path))
        if task == "cls":
            y = y.float()
        elif task == "reg":
            y = y.float()
        else:
            raise ValueError("task should be reg or cls")
        self.label = y


    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        sc = self.sc[idx]         # [N, N]
        fc = self.fc[idx]         # [N, N]
        xs = self.sc[idx]
        xf = self.fc[idx]
        label = self.label[idx]
        return sc, fc, xs, xf, label

    def Get_Index(self):
        idx = list(range(len(self.label)))
        return idx
    

