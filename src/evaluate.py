import torch
from torch.utils.data import DataLoader
from model.neural_net import Model
from torch.nn.functional import softmax


def evaluate(model: Model, data_loader: DataLoader):
    if data_loader.batch_size != 1:
        return "data loader batch size must be 1"
    # TODO: modernize function
    for i_image, image in enumerate(data_loader):
        logits = model(image['image'])
        print(f"************ {i_image} ************\n"
              f"{softmax(logits)}\n"
              f"prediction - {torch.max(logits)[1]}, actually {image['labels']}")
