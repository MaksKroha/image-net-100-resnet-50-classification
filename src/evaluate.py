import time

import torch
from torch.utils.data import DataLoader
from src.model.neural_net import Model
from torch.nn.functional import softmax

from src.utils.analyzer import Analyzer
from src.utils.logger import exception_logger
import matplotlib.pyplot as plt

def get_label_prob(model, tensor):
    logits = model(tensor)
    max_val, max_index = torch.max(softmax(logits, dim=1), dim=1)
    return max_index.item(), max_val.item()

def get_name_from_label(label, json_data):
    for value in json_data.values():
        if value["index"] == label:
            return value["name"]
    return None

@exception_logger
def evaluate(model: Model,
             data_loader: DataLoader,
             json_data,
             eval_mode: int):
    print("------Evaluating model------")

    model = model.to("cpu")
    model.eval()
    if data_loader.batch_size != 1:
        raise RuntimeError('Batch size must be 1')

    if eval_mode == 0:
        for i_image, image in enumerate(data_loader):
            img = image["image"].squeeze(0).permute(1, 2, 0)
            img = img * 255.0
            if img.dtype != torch.uint8:
                img = img.to(torch.uint8)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            pred_label, prob = get_label_prob(model, image["image"])
            print(f"-- {pred_label} {prob}%, truth - {image["labels"]},"
                  f" {get_name_from_label(pred_label, json_data)}")
            time.sleep(10)
    elif eval_mode == 1:
        analyzer = Analyzer("images number",
                            "probability",
                            "Evaluating accuracy(probabilities) graph",
                            "",
                            "green")
        for i_image, image in enumerate(data_loader):
            print(image["image"])
            pred_label, prob = get_label_prob(model, image["image"])
            analyzer.add_test_val(prob * 100 if pred_label == image["labels"] else 0.0, i_image)
            analyzer.show_accuracy()
    return 0
