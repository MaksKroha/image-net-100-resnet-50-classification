import torch
from torch.utils.data import DataLoader
from src.model.neural_net import Model
from src.backward import backward
from src.utils.analizer import Analizer

import time


def train(model: Model, data_loader: DataLoader, epochs: int,
          device: str, lr: float, t_max: int,
          lr_min: float, weight_decay: float,
          analizer=None):
    # model - neural net model to train
    # data_loader - data loader with pin_memory=device
    # epochs - the num of epochs
    # device - cpu or cuda
    # lr - learning rate
    # t_max - lr decreasing epochs (for lr scheduler)
    # lr_min - lr at the end of decreasing (lr scheduler)
    # weight_decay - fine for grads in weight decay regularization

    if device == "cuda" and not torch.cuda.is_available():
        return "Cuda is not available"
    start = time.time()
    model.train()  # enable train mode
    end = time.time()
    print(f"Model enabling train mode - {end - start}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr_min)
    
    for epoch in range(epochs):
        start = time.time()
        for i, batch in enumerate(data_loader):
            end = time.time()
            print(f"{i} batch loading time - {end - start}")
            
            start = time.time()
            batch['labels'] = batch['labels'].to(device)
            batch['image'] = batch['image'].to(device)
            end = time.time()
            print(f"Converting batch to {device} - {end - start}")

            start = time.time()
            logits = model(batch['image'])
            end = time.time()
            print(f"Forward pass - {end - start}")

            start = time.time()
            backward(logits, batch['labels'], optim, analizer=analizer)
            end = time.time()
            print(f"Backpropagation time - {end - start}")
            scheduler.step()
            start = time.time()

