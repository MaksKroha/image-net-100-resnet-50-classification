import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from skimage import io
import numpy as np


class ImageNetDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        print(csv_file)
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return self.landmarks_frame.dropna(how='all').shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.landmarks_frame.iloc[idx, 0])

        label = self.landmarks_frame.iloc[idx, 1]
        label = torch.tensor(label, dtype=torch.long)
        sample = {'image': image, 'labels': label}

        if self.transform:
            sample["image"] = self.transform(image)

        return sample