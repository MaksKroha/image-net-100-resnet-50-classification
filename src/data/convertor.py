import io
import lmdb
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import json

from src.utils.logger import exception_logger


class LMDB:
    def __init__(self, db_path, json_file, map_size=53687091200):
        self.json_data = json.load(open(json_file))
        self.env = lmdb.open(db_path, map_size=map_size)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor()
        ])

        # Creating directory if not exists
        Path(db_path).mkdir(parents=True, exist_ok=True)

    @exception_logger
    def get_label(self, image_path):
        if len(image_path) >= 9 and image_path[:9] in self.json_data:
            return self.json_data[image_path[:9]]["index"]
        else:
            raise NameError("Bad jpeg file name (there is no nxxxxxxxx)")


    @exception_logger
    def __put_image(self, txn, folder_path, img_name, current_idx, label):
        if not (img_name.lower().endswith(".jpeg") or img_name.lower().endswith(".jpg")):
            return None

        with Image.open(f"{folder_path}/{img_name}") as img:
            img = img.convert('RGB')
            tensor = self.transform(img)

            if tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)

            label = self.get_label(img_name) if label is None else label

            key = f"{current_idx:010d}".encode()
            value = (tensor, label)

            buffer = io.BytesIO()
            torch.save(value, buffer)

            txn.put(key, buffer.getvalue())
        return 0


    @exception_logger
    def add_images_from_folder(self, folder_path, label=None):
        """
        Add all images from a folder to LMDB
        """

        image_names = [image for image in os.listdir(folder_path)]

        with self.env.begin(write=True) as txn:
            current_idx = self.get_last_index(txn)
            if current_idx is None:
                raise IndexError("LMDB wrong index")

            for img_name in image_names:
                if self.__put_image(txn, folder_path, img_name, current_idx, label) is not None:
                    current_idx += 1

            txn.put(b'__current_index__', str(current_idx).encode())
        return 0


    @exception_logger
    def get_last_index(self, txn):
        """
        return a last index in LMDB
        """
        current_idx_bytes = txn.get(b'__current_index__')
        return int(current_idx_bytes.decode()) if current_idx_bytes else 0

    @exception_logger
    def close(self):
        """
        closing connection with database
        """
        self.env.close()

    @exception_logger
    def get_by_index(self, index):
        key = f"{index:010d}".encode()
        with self.env.begin(write=False) as txn:
            value = txn.get(key)

            buffer = io.BytesIO(value)
            tensor, label = torch.load(buffer)
            return tensor, label
