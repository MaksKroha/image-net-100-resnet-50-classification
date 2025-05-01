from dotenv import load_dotenv
from os import getenv

from src.utils.logger import exception_logger
from src.utils.timer import timed

load_dotenv()
# setting global variables
LOG_FILE=getenv('LOG_FILE')


# Setting settings for logining file
import logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d (%(funcName)s) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

import torch
import time
from torch.utils.data import DataLoader

from src.data.custom_dataset import ImageNetDataset
from src.evaluate import evaluate
from src.model.neural_net import Model

from src.train import train
from src.utils.analyzer import Analyzer
from src.data.convertor import LMDB

import sys

@exception_logger
def main(*args, **kwargs):
    lmdb_path = "/home/maksymkroha/MineFiles/kaggle/image-net-100/lmdb"
    labels_json = "/home/maksymkroha/MineFiles/imageNetResNetClassification/dataset/labels.json"
    trained_model = "/home/maksymkroha/MineFiles/imageNetResNetClassification/models/trained_model.pt"

    lmdb_db = LMDB(lmdb_path, labels_json)
    
    with lmdb_db.env.begin(write=False) as txn:
        lmdb_length = lmdb_db.get_last_index(txn)
        if lmdb_length is None:
            print("Wrong value for lmdb length", file=sys.stderr)
            return 1


    epochs = 5
    t_max = 1
    lr = 1e-4/2
    lr_min = 1e-4/2
    weight_decay = 5e-5
    device = "cpu"
    output_classes = 100
    batch_size = 128
    num_workers = 1
    prefetch_factor = 2

    print(f"-- learning rate - {lr}")
    print(f"-- lr_min - {lr_min}")
    print(f"-- weight decay - {weight_decay}")
    time.sleep(5)

    model = Model(output_classes)
    analyzer = Analyzer()

    exception_logger(timed(model.load_state_dict))(torch.load(trained_model,
                                                              weights_only=True,
                                                              map_location=device))


    # Dataset configuration
    print("---- Dataloader/dataloader configure ------")
    dataset = ImageNetDataset(lmdb_db, lmdb_length)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True,
                            pin_memory=True, num_workers=num_workers,
                            prefetch_factor=prefetch_factor)

    # training
    print(f"---- Train starts")
    model = exception_logger(timed(model.to))(device)

    if train(model, dataloader, epochs, device, lr, t_max, lr_min, weight_decay, analyzer) is None:
        print("---- Train ends with exception ", file=sys.stderr)
    
    exception_logger(timed(torch.save))(model.state_dict(), trained_model)

    # evaluation
    model = exception_logger(timed(model.to))("cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    timed(evaluate(model, dataloader, 100))


if __name__ == "__main__":
    main()