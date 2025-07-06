from dotenv import load_dotenv
from os import getenv

from src.utils.logger import exception_logger
from src.utils.timer import timed

load_dotenv()
# setting global variables
TRAIN_LMDB_PATH: str = getenv("TRAIN_LMDB_PATH")
TEST_LMDB_PATH: str = getenv("TEST_LMDB_PATH")
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
EVAL_MODE: int = int(getenv('EVAL_MODE'))
DROPOUT: float = float(getenv('DROPOUT'))
STOCHASTIC_DEPTH_P: float = float(getenv('STOCHASTIC_DEPTH_P'))

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
from src.model.neural_net import Model

from src.train import train
from src.utils.analyzer import Analyzer
from src.data.convertor import LMDB

import json

@exception_logger
def main_train(*args, **kwargs):
    train_lmdb_db = LMDB(TRAIN_LMDB_PATH, LABELS_JSON_PATH)
    test_lmdb_db = LMDB(TEST_LMDB_PATH, LABELS_JSON_PATH)
    json_data = json.load(open(LABELS_JSON_PATH))

    with train_lmdb_db.env.begin(write=False) as txn:
        train_lmdb_length = train_lmdb_db.get_last_index(txn)
    with test_lmdb_db.env.begin(write=False) as txn:
        test_lmdb_length = test_lmdb_db.get_last_index(txn)

    print(f"train_lmdb_length: {train_lmdb_length}")
    print(f"test_lmdb_length: {test_lmdb_length}")

    print(f"-- learning rate - {LR}")
    print(f"-- lr_min - {LR_MIN}")
    print(f"-- weight decay - {WEIGHT_DECAY}")
    time.sleep(5)

    model = Model(OUTPUT_CLASSES, DROPOUT, STOCHASTIC_DEPTH_P)
    analyzer = Analyzer("epochs number",
                        "mean loss",
                        "Training loss curve",
                        "red",
                        "green")
    model = Model(OUTPUT_CLASSES, DROPOUT, WEIGHT_DECAY)

    # print("---- Loading trained model ----")
    # exception_logger(timed(model.load_state_dict))(torch.load(TRAINED_MODEL_PATH,
                                                              # weights_only=True,
                                                              # map_location=DEVICE))

    print("---- Dataset/dataloader configure ------")
    train_dataset = ImageNetDataset(train_lmdb_db, train_lmdb_length)
    test_dataset = ImageNetDataset(test_lmdb_db, test_lmdb_length)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=True,
                                  num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True,
                                 num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR)

    # training
    print(f"---- Train starts")
    train(model, train_dataloader, test_dataloader, EPOCHS, DEVICE, LR, T_MAX, LR_MIN,
          WEIGHT_DECAY, analyzer)

    exception_logger(timed(torch.save))(model.state_dict(), TRAINED_MODEL_PATH)

    return 0

if __name__ == "__main__":
    main_train()
