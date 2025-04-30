import io
import lmdb
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import json


class LMDB:
    def __init__(self, db_path, json_file, map_size=53687091200):
        self.json_data = json.load(open(json_file))
        self.env = lmdb.open(db_path, map_size=map_size)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor()
        ])

        # Створюємо папку, якщо не існує
        Path(db_path).mkdir(parents=True, exist_ok=True)

    def get_label(self, image_path):
        """
        Витягує мітку з імені файлу (перед першим символом '_')
        Приклад: "cat_123.jpg" -> "cat"
        """
        if len(image_path) > 10 and image_path[:9] in self.json_data:
            return self.json_data[image_path[:9]]["index"]
        else:
            raise NameError("Bad jpeg file name (there is no nxxxxxxxx)")

    def add_images_from_folder(self, folder_path):
        """
        Додає всі JPEG зображення з папки до LMDB бази
        """

        image_paths = [image for image in os.listdir(folder_path)]

        with self.env.begin(write=True) as txn:
            current_idx = self._get_last_index(txn)

            for img_path in image_paths:
                try:
                    # Відкриваємо та обробляємо зображення
                    if not (img_path.lower().endswith(".jpeg") or img_path.lower().endswith(".jpg")):
                        continue

                    with Image.open(f"{folder_path}/{img_path}") as img:
                        img = img.convert('RGB')
                        tensor = self.transform(img)

                        if tensor.dtype != torch.uint8:
                            tensor = tensor.to(torch.uint8)

                        # Отримуємо мітку
                        label = self.get_label(img_path)

                        # Готуємо дані для запису
                        new_idx = current_idx + 1
                        key = f"{current_idx:010d}".encode()
                        value = (tensor, label)

                        # Серіалізація
                        buffer = io.BytesIO()
                        torch.save(value, buffer)

                        # Запис у базу
                        txn.put(key, buffer.getvalue())

                        current_idx = new_idx
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            # Оновлюємо лічильник індексів
            txn.put(b'__current_index__', str(current_idx).encode())

    def _get_last_index(self, txn):
        """
        Повертає останній використаний індекс з бази
        """
        current_idx_bytes = txn.get(b'__current_index__')
        return int(current_idx_bytes.decode()) if current_idx_bytes else 0

    def close(self):
        """
        Закриває з'єднання з базою
        """
        self.env.close()

    def get_by_index(self, index):      
      key = f"{index:010d}".encode()  # Форматуємо ключ як 10-значний рядок з ведучими нулями
      with self.env.begin(write=False) as txn:
          value = txn.get(key)
          
          # Десеріалізація даних
          buffer = io.BytesIO(value)
          tensor, label = torch.load(buffer)
          return tensor, label

# Приклад використання
if __name__ == "__main__":
    inserter = LMDBInserter("/home/maksymkroha/MineFiles/kaggle/image-net-100/lmdb"
                            , "/home/maksymkroha/MineFiles/imageNetResNetClassification/dataset/labels.json")
    json_data = inserter.json_data

    root_dir = r"/home/maksymkroha/MineFiles/kaggle/image-net-100"
    for folder1 in os.listdir(root_dir):
        if len(folder1) > 5 and folder1.startswith("train"):
            for folder2 in os.listdir(f"{root_dir}/{folder1}"):
                print(f"processing - {root_dir}/{folder1}/{folder2}")
                inserter.add_images_from_folder(f"{root_dir}/{folder1}/{folder2}")

    inserter.view_lmdb_content(100)
    inserter.close()