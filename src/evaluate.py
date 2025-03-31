import torch
from torch.utils.data import DataLoader
from src.model.neural_net import Model
from torch.nn.functional import softmax


def evaluate(model: Model, data_loader: DataLoader, test_img_num):
    print("Evaluating model")

    model.eval()
    if data_loader.batch_size != 1:
        return "der batch size must be 1"

    for i_image, image in enumerate(data_loader):
        logits = model(image['image'])
        print(f"************ {i_image} ************\n"
              f"{softmax(logits, dim=1)}\n"
              f"prediction - {torch.argmax(logits).item()}, actually {image['labels']}")
        print(f"logits {logits}")

        if i_image >= test_img_num:
            break
