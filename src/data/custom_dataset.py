import torch
from torch.utils.data import Dataset
import lmdb

import time 
class ImageNetDataset(Dataset):
    def __init__(self, lmdb, lmdb_length, transform=None):
        self.lmdb = lmdb
        self.lmdb_length = lmdb_length

    def __len__(self):
        return self.lmdb_length

    def __getitem__(self, idx):
        res = self.lmdb.get_by_index(idx)
        if res is None:
          raise Exception(f"Indx - {idx}")
        image, label = res
        image = image.to(torch.float32) / 255 # normalization
        sample = {'image': image, 'labels': label}
        return {'image': image, 'labels': label}