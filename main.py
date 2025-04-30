# Setting settings for logining file
import logging

from torch.utils import data
logging.basicConfig(
    filename='logs.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d (%(funcName)s) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import torch
import time
from torch.utils.data import DataLoader
from torch.xpu import device

from src.data.custom_dataset import ImageNetDataset
from src.evaluate import evaluate
from src.model.neural_net import Model
from torchvision import transforms

from src.train import train
from src.utils.formatter import format_paths_into_csv_name_label
from src.utils.analizer import Analizer
from src.data.convertor import LMDB

import os 
from google.colab import drive

if __name__ == "__main__":
    lmdb_path = "/content/lmdb"
    labels_json = "/content/drive/MyDrive/python_projects/imageNetResNetClassification/dataset/labels.json"

    lmdb_db = LMDB(lmdb_path, labels_json)
    
    with lmdb_db.env.begin(write=False) as txn:
        lmdb_length = lmdb_db._get_last_index(txn) 
    print(f"lmdb length - {lmdb_length}")
    trained_model = "/content/drive/MyDrive/python_projects/imageNetResNetClassification/models/trained_model.pt"

    epochs = 5
    t_max = 1
    lr = 1e-4/2
    lr_min = 1e-4/2
    weight_decay = 5e-5
    device = "cuda"
    output_clases = 100 # cats - 0, dogs - 1

    print(f"lr - {lr}")
    print(f"lr_min - {lr_min}")
    print(f"weight decay - {weight_decay}")
    time.sleep(5)

    model = Model(output_clases)
    analizer = Analizer()

    try:
        model.load_state_dict(torch.load(trained_model, weights_only=True))
    except Exception as e:
        print(e)
        time.sleep(5)


    # Dataset configuration
    print("Dataloader/dataloader configure")
    dataset = ImageNetDataset(lmdb_db, lmdb_length)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=True, drop_last=True,
                            pin_memory=True, num_workers=1,
                            prefetch_factor=2)

    # training
    print(f"Train strarts")
    start = time.time()
    model = model.to(device)
    end = time.time()
    print(f"Model to {device} - {end - start}")
    train(model, dataloader, epochs, device, lr, t_max, lr_min, weight_decay, analizer)
    
    start = time.time()
    torch.save(model.state_dict(), trained_model)
    end = time.time()
    print(f"Torch saving model - {end - start}")

    # evaluation
    model = model.to("cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    evaluate(model, dataloader, 100)
