from dotenv import load_dotenv
from os import getenv

from src.utils.logger import exception_logger
from src.utils.timer import timed

load_dotenv()
# setting global variables
LMDB_PATH: str = getenv("LMDB_PATH")
LABELS_JSON_PATH: str = getenv("LABELS_JSON_PATH")
TRAINED_MODEL_PATH: str = getenv("TRAINED_MODEL_PATH")

LOG_FILE: str = getenv('LOG_FILE')
EPOCHS: int = int(getenv('EPOCHS'))
T_MAX: int = int(getenv('T_MAX'))
LR: float = float(getenv('LR'))
LR_MIN: float = float(getenv('LR_MIN'))
WEIGHT_DECAY: float = float(getenv('WEIGHT_DECAY'))
DEVICE: str = getenv('DEVICE')
OUTPUT_CLASSES: int = int(getenv('OUTPUT_CLASSES'))
BATCH_SIZE: int = int(getenv('BATCH_SIZE'))
NUM_WORKERS: int = int(getenv('NUM_WORKERS'))
PREFETCH_FACTOR: int = int(getenv('PREFETCH_FACTOR'))

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
    lmdb_db = LMDB(LMDB_PATH, LABELS_JSON_PATH)

    with lmdb_db.env.begin(write=False) as txn:
        lmdb_length = lmdb_db.get_last_index(txn)
        if lmdb_length is None:
            raise RuntimeError(f"LMDB length is None")

    print(f"-- learning rate - {LR}")
    print(f"-- lr_min - {LR_MIN}")
    print(f"-- weight decay - {WEIGHT_DECAY}")
    time.sleep(5)

    model = Model(OUTPUT_CLASSES)
    analyzer = Analyzer()

    exception_logger(timed(model.load_state_dict))(torch.load(TRAINED_MODEL_PATH,
                                                              weights_only=True,
                                                              map_location=DEVICE))

    # Dataset configuration
    print("---- Dataloader/dataloader configure ------")
    dataset = ImageNetDataset(lmdb_db, lmdb_length)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, drop_last=True,
                            pin_memory=True, num_workers=NUM_WORKERS,
                            prefetch_factor=PREFETCH_FACTOR)

    # training
    print(f"---- Train starts")
    model = exception_logger(timed(model.to))(DEVICE)

    if train(model, dataloader, EPOCHS, DEVICE, LR, T_MAX, LR_MIN,
             WEIGHT_DECAY, analyzer) is None:
        print("---- Train ends with exception ", file=sys.stderr)

    exception_logger(timed(torch.save))(model.state_dict(), TRAINED_MODEL_PATH)

    # evaluation
    model = exception_logger(timed(model.to))("cpu")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    timed(evaluate(model, dataloader, 10))

    return 0

if __name__ == "__main__":
    if main() is None:
        print("During executing main errors occurs", file=sys.stderr)
