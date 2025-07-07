import torch
from torch.utils.data import DataLoader
from src.model.neural_net import Model
from src.backward import backward
from src.utils.logger import exception_logger
from src.utils.timer import timed


@exception_logger
def learning_cycle(model, optim, device, analyzer, batch):
    batch['labels'] = batch['labels'].to(device)
    batch['image'] = batch['image'].to(device)

    logits = model(batch['image'])

    backward(logits, batch['labels'], optim, analyzer=analyzer)


@exception_logger
def train(model: Model, train_data_loader: DataLoader,
          test_dataloader: DataLoader, epochs: int,
          device: str, lr: float, t_max: int,
          lr_min: float, weight_decay: float,
          analyzer=None, trained_model_path: str = None):
    # model - neural net model to train
    # data_loader - data loader with pin_memory=device
    # epochs - the num of epochs
    # device - cpu or cuda
    # lr - learning rate
    # t_max - lr decreasing epochs (for lr scheduler)
    # lr_min - lr at the end of decreasing (lr scheduler)
    # weight_decay - fine for grads in weight decay regularization

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    model = model.to(device)
    model.train()  # enable train mode

    optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, eta_min=lr_min)

    for epoch in range(epochs):
        for i, batch in enumerate(train_data_loader):
            learning_cycle(model, optim, device, analyzer, batch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                logits = model(batch['image'])
                loss = criterion(logits, batch['labels'])
                analyzer.add_test_val(loss.item())
        model.train()

        torch.save(model.state_dict(), trained_model_path)
        analyzer.show_accuracy()
    return 0