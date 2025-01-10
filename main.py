import csv
import torch
from DataStructuring import DataStruct
from Model import Model
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # hyperparameters and main variables
    # lr_list = [0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
    lr = 0.01
    blocks_num = 16
    epochs = 1
    weight_decay = 0.0005
    batch_size = 64
    test_images_num = 10
    trained_classes_num = 120
    train_mode = True
    test_mode = True
    load_previous_state_dict = False
    shuffle = False  # --temp
    print_loss = True
    train_device = "cuda"
    images_path_int_file = "images_paths.csv"
    model = Model()
    model.setBlocksProbabilities(blocks_num)

    if load_previous_state_dict:
        try:
            state_dict = torch.load("parameters/model_state_dict.pt", weights_only=True)
            model.load_state_dict(state_dict)
            print("-- Loaded state dict ")
        except(FileNotFoundError, EOFError):
            pass

    image_transform = transforms.Compose([
        transforms.ToTensor(),  # make pixels from range [0, 255] to [0, 1]
        transforms.Resize((224, 224)),
        # Normalize class do not count mean and std vals by itself
        # we just indicates those values in parameters
        # Those vals is developed by time (experience)
        transforms.RandomHorizontalFlip(p=0.5)  # 50% probability to flip
        # contrast it is difference between bright and dark pixels
        # range of contrast constant is [1 - 0.5,  1 + 0.5]
    ])
    if train_mode:
        # print("-- Shuffle images_paths_file 3 times")
        # DataStruct.shuffle_csv_file_by_rows(images_path_int_file, 3)

        print("-- Deep Learning starts")
        model = model.to(train_device)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            print(f"- {epoch + 1}/{epochs} epoch started")

            # for param_group in optim.param_groups:
            #     param_group["lr"] = lr_list[epoch]
            # print(f"- current learning rate = {lr_list[epoch]}")

            if shuffle:
                print("- shuffle images_paths_file 1 times")
                DataStruct.shuffle_csv_file_by_rows(images_path_int_file, 1)

            with open(images_path_int_file, "r", newline="\n") as images_path_int_file_thread:
                reader = csv.reader(images_path_int_file_thread, delimiter=",")
                tensor, labels = None, []

                for i, row in enumerate(reader):
                    if tensor is None:
                        tensor = image_transform(Image.open(row[0])).unsqueeze(0)
                    else:
                        tensor = torch.cat([tensor, image_transform(Image.open(row[0])).unsqueeze(0)], dim=0)

                    labels.append(int(row[1]))

                    if i % batch_size == batch_size - 1:
                        tensor, labels = tensor.to(train_device), torch.tensor(labels).to(train_device)
                        model.backward(model(tensor), labels, optim, print_loss)
                        tensor, labels = None, []
                        print("+1", end="\n")  # +1 batch

                if tensor is not None:
                    tensor, labels = tensor.to(train_device), torch.tensor(labels).to(train_device)
                    model.backward(model(tensor), labels, optim)

        print("-- Saving model parameters")
        torch.save(model.state_dict(), "parameters/model_state_dict.pt")

    if test_mode:
        test_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

        model.eval()
        model.to("cpu")

        print("Opening image paths file")
        with open(images_path_int_file, "r", newline="\n") as images_path_int_file_thread:
            reader = csv.reader(images_path_int_file_thread, delimiter=",")
            for i, row in enumerate(reader):
                if i == test_images_num: break
                print(f"------------------{i + 1}-------------------")
                image = test_image_transform(Image.open(row[0]))
                with torch.no_grad():
                    logits = model(image.unsqueeze(0))
                    prob = torch.nn.functional.softmax(logits, dim=1)
                    print(f"probs - {prob[0][0: trained_classes_num]}")
                    print(f"max (prob, index) = ({torch.max(prob, dim=1)})")
                    print(f"right class = {row[1]}")

                transformed_image_np = image.permute(1, 2, 0).numpy()
                plt.imshow(transformed_image_np)
                plt.axis("off")
                plt.show()