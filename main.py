import csv
import torch
from torchvision import transforms
from PIL import Image


if __name__ == "__main__":
    # hyperparameters and main variables
    epochs = 1
    batch_size = 128
    image_paths_file = "images_paths.csv"
    images_num = 120*100
    image_transform = transforms.Compose([
        transforms.ToTensor(),  # make pixels from range [0, 255] to [0, 1]
        transforms.Resize((224, 224)),
        # Normalize class do not count mean and std vals by itself
        # we just indicates those values in parameters
        # Those vals is developed by time (experience)
        transforms.Normalize([0.5, 0.5, 0.4], [0.23, 0.23, 0.23])
    ])

    # DataLoader.shuffle_csv_file_by_rows(image_paths_file, images_num, 3)
    for _ in range(epochs):
        # DataLoader.shuffle_csv_file_by_rows(image_paths_file, images_num, 1)
        with open(image_paths_file, "r", newline="\n") as image_paths_file_thread:
            reader = csv.reader(image_paths_file_thread, delimiter=",")
            tensor = None
            labels = []
            for i, row in enumerate(reader):
                if tensor is not None:
                    tensor = torch.cat([tensor, image_transform(Image.open(row[0])).unsqueeze(0)], dim=0)
                else:
                    tensor = image_transform(Image.open(row[0])).unsqueeze(0)
                labels.append(int(row[1]))
                if i % batch_size == batch_size - 1:
                    print("some function to batch")
                    tensor = None
                    labels = []
            if tensor is not None:
                print("some function to not full batch")
