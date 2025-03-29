import torch
from torch.utils.data import DataLoader
from model.neural_net import Model
from src.backward import backward


def train(model: Model, data_loader: DataLoader, epochs: int,
          device: str, lr: float, t_max: int,
          lr_min: float, weight_decay: float):
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

    model = model.to(device)
    model.train()  # enable train mode

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr_min)

    for epoch in range(epochs):
        for batch in data_loader:
            logits = model(batch['image'])
            backward(logits, batch['labels'], optim)

        scheduler.step()
