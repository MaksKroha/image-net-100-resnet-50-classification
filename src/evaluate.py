import torch
from torch.utils.data import DataLoader
from src.model.neural_net import Model
from torch.nn.functional import softmax

from src.utils.logger import exception_logger


@exception_logger
def evaluate(model: Model, data_loader: DataLoader, test_img_num):
    print("------Evaluating model------")

    model = model.to("cpu")
    model.eval()
    if data_loader.batch_size != 1:
        raise RuntimeError('Batch size must be 1')

    for i_image, image in enumerate(data_loader):
        logits = model(image['image'])
        max_val, max_index = torch.max(softmax(logits, dim=1), dim=1)
        max_val = max_val.item()
        max_index = max_index.item()
        print(f"************ {i_image} ************\n"
              f"prediction - {max_index} {max_val}%, actually {image['labels']}")

        if i_image >= test_img_num:
            break
