import csv
import torch
import matplotlib

from my_data.DataStructuring import DataStruct
from Model import Model
from torchvision import transforms
from PIL import Image

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # hyperparameters and main variables
    lr = 0.001
    # For cosine lr scheduler
    min_lr = 0.00001
    lr_decreasing_periods = 3

    res_blocks_num = 16
    epochs = 3
    weight_decay = 0.00001
    batch_size = 64
    test_images_num = 10  # Top Boarder
    trained_classes_num = 25
    p_last_sth_depth = 0.6  # last block prob in stochastic depth
    train_mode = False
    test_mode = True
    load_previous_state_dict = True
    shuffle = True  # --temp
    print_loss = True
    train_device = "cuda"

    images_path_int_file = "csvs/25_dif_classes_images_path.csv"

    model = Model()
    model.setBlocksProbabilities(res_blocks_num, p_last_sth_depth)
    torch.set_default_dtype(torch.float32)

    if load_previous_state_dict:
        try:
            state_dict = torch.load("parameters/model_state_dict.pt", weights_only=True)
            model.load_state_dict(state_dict)
            print("-- Loaded previous state dict ")
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
        model = model.to(train_device)

        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=lr_decreasing_periods, eta_min=min_lr)

        for epoch in range(epochs):
            print(f"- {epoch + 1}/{epochs} epoch started")

            DataStruct.shuffle_csv_file_by_rows(images_path_int_file, 1)
            print("Shuffled images_path_int file")

            with open(images_path_int_file, "r", newline="\n") as images_path_int_file_thread:
                reader = csv.reader(images_path_int_file_thread, delimiter=",")
                tensor, labels = None, []

                for i, row in enumerate(reader):
                    if tensor is None:
                        tensor = image_transform(Image.open(row[0])).unsqueeze(0)
                        if tensor.shape[1] == 1:
                            tensor = tensor.repeat(1, 3, 1, 1)
                    else:
                        new_tensor = image_transform(Image.open(row[0])).unsqueeze(0)
                        if new_tensor.shape[1] == 1:
                            new_tensor = new_tensor.repeat(1, 3, 1, 1)
                        tensor = torch.cat([tensor, new_tensor], dim=0)

                    # # LABEL SMOOTHING
                    # label = torch.zeros(trained_classes_num)
                    # label[int(row[1])] = 1
                    # labels.append(((1 - lb_epsi) * label + lb_epsi / (trained_classes_num - 1)).round(decimals=3))
                    labels.append(int(row[1]))

                    if i % batch_size == batch_size - 1:
                        tensor, labels = tensor.to(train_device), torch.tensor(labels).to(train_device)
                        model.backward(model(tensor), labels, optim, print_loss)
                        tensor, labels = None, []
                        print("+1", end=" ")  # +1 batch

                if tensor is not None:
                    tensor, labels = tensor.to(train_device), torch.tensor(labels).to(train_device)
                    model.backward(model(tensor), labels, optim)
            scheduler.step()

        torch.save(model.state_dict(), "parameters/model_state_dict.pt")
        print("-- Saved model parameters")

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
                    print(f"max (prob, index) = ({torch.max(prob, dim=1)})", end="   ")
                    print(f"right class = {row[1]}")

                transformed_image_np = image.permute(1, 2, 0).numpy()
                plt.imshow(transformed_image_np)
                plt.axis("off")
                plt.show()
