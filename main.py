import torch
from torch.utils.data import DataLoader
from torch.xpu import device

from src.data.custom_dataset import ImageNetDataset
from src.evaluate import evaluate
from src.model.neural_net import Model
from torchvision import transforms

from src.train import train
from src.utils.formatter import format_paths_into_csv_name_label

if __name__ == "__main__":
    # cats - 0
    # dogs - 1
    # format_paths_into_csv_name_label(r"/home/maksymkroha/MineFiles/imageNetResNetClassification/data/archive/PetImages/Cat",
    #                                  r"/home/maksymkroha/MineFiles/imageNetResNetClassification/data/images.csv",
    #                                  100, 0)
    # format_paths_into_csv_name_label(
    #     r"/home/maksymkroha/MineFiles/imageNetResNetClassification/data/archive/PetImages/Animals",
    #     r"/home/maksymkroha/MineFiles/imageNetResNetClassification/data/images.csv",
    #     200)
    epochs = 10
    t_max = 8
    lr = 1e-5
    lr_min = 1e-6
    weight_decay = 1e-6
    device = "cpu"
    output_clases = 2 # cats - 0, dogs - 1

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    # csv file with line in style "path, label"
    csv_file = "/datasets/images.csv"

    model = Model(output_clases)
    try:
        model.load_state_dict(torch.load("models/trained_model.pt", weights_only=True))
    except Exception as e:
        print(e)


    # Dataset configuration
    dataset = ImageNetDataset(csv_file, transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # training
    train(model, dataloader, epochs, device, lr, t_max, lr_min, weight_decay)
    torch.save(model.state_dict(), "models/trained_model.pt")

    # evaluation
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    evaluate(model, dataloader, 3)
