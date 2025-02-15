import torch
from torch import nn
from sklearn.metrics import accuracy_score
import numpy as np


class Model(nn.Module):
    def __init__(self):
        torch.set_default_dtype(torch.float32)
        self.block_probs = []
        super(Model, self).__init__()

        # conv1_x
        self.main_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.main_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # realized bottleneck blocks in "full pre-activation" style
        self.conv2_x = nn.ModuleList([])
        self.skip_con2_x = nn.ModuleList([])
        input_size = 64
        blocks_quality = 3
        for i in range(blocks_quality):
            self.skip_con2_x.append(nn.Conv2d(input_size, 256, kernel_size=1, bias=False))

            self.conv2_x.append(nn.ModuleList([]))
            self.conv2_x[i].append(nn.Conv2d(input_size, 64, kernel_size=1, bias=False))
            self.conv2_x[i].append(nn.BatchNorm2d(64))
            self.conv2_x[i].append(nn.ReLU(inplace=True))

            self.conv2_x[i].append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            self.conv2_x[i].append(nn.BatchNorm2d(64))
            self.conv2_x[i].append(nn.ReLU(inplace=True))

            self.conv2_x[i].append(nn.Conv2d(64, 256, kernel_size=1, bias=False))
            self.conv2_x[i].append(nn.BatchNorm2d(256))
            input_size = 256

        self.conv3_x = nn.ModuleList([])
        self.skip_con3_x = nn.ModuleList([])
        blocks_quality = 4
        for i in range(blocks_quality):
            self.skip_con3_x.append(nn.Conv2d(input_size, 512, kernel_size=1, bias=False))

            self.conv3_x.append(nn.ModuleList([]))
            self.conv3_x[i].append(nn.Conv2d(input_size, 128, kernel_size=1, bias=False))
            self.conv3_x[i].append(nn.BatchNorm2d(128))
            self.conv3_x[i].append(nn.ReLU(inplace=True))

            self.conv3_x[i].append(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False))
            self.conv3_x[i].append(nn.BatchNorm2d(128))
            self.conv3_x[i].append(nn.ReLU(inplace=True))

            self.conv3_x[i].append(nn.Conv2d(128, 512, kernel_size=1, bias=False))
            self.conv3_x[i].append(nn.BatchNorm2d(512))
            input_size = 512
        self.skip_con3_x[0].stride = 2
        self.conv3_x[0][0].stride = 2


        self.conv4_x = nn.ModuleList([])
        self.skip_con4_x = nn.ModuleList([])
        blocks_quality = 6
        for i in range(blocks_quality):
            self.skip_con4_x.append(nn.Conv2d(input_size, 1024, kernel_size=1, bias=False))

            self.conv4_x.append(nn.ModuleList([]))
            self.conv4_x[i].append(nn.Conv2d(input_size, 256, kernel_size=1, bias=False))
            self.conv4_x[i].append(nn.BatchNorm2d(256))
            self.conv4_x[i].append(nn.ReLU(inplace=True))

            self.conv4_x[i].append(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
            self.conv4_x[i].append(nn.BatchNorm2d(256))
            self.conv4_x[i].append(nn.ReLU(inplace=True))

            self.conv4_x[i].append(nn.Conv2d(256, 1024, kernel_size=1, bias=False))
            self.conv4_x[i].append(nn.BatchNorm2d(1024))
            input_size = 1024
        self.skip_con4_x[0].stride = 2
        self.conv4_x[0][0].stride = 2

        self.conv5_x = nn.ModuleList([])
        self.skip_con5_x = nn.ModuleList([])
        blocks_quality = 3
        for i in range(blocks_quality):
            self.skip_con5_x.append(nn.Conv2d(input_size, 2048, kernel_size=1, bias=False))

            self.conv5_x.append(nn.ModuleList([]))
            self.conv5_x[i].append(nn.Conv2d(input_size, 512, kernel_size=1, bias=False))
            self.conv5_x[i].append(nn.BatchNorm2d(512))
            self.conv5_x[i].append(nn.ReLU(inplace=True))

            self.conv5_x[i].append(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False))
            self.conv5_x[i].append(nn.BatchNorm2d(512))
            self.conv5_x[i].append(nn.ReLU(inplace=True))

            self.conv5_x[i].append(nn.Conv2d(512, 2048, kernel_size=1, bias=False))
            self.conv5_x[i].append(nn.BatchNorm2d(2048))
            input_size = 2048
        self.skip_con5_x[0].stride = 2
        self.conv5_x[0][0].stride = 2

        # Global average pooling

        self.fcn = nn.Linear(2048, 120)

    def forward(self, images):
        blocks_remain = np.random.binomial(1, self.block_probs) if self.training else self.block_probs
        print(blocks_remain)
        tensor = self.main_conv(images)
        tensor = self.main_max_pool(tensor)

        blocks_count = 0
        for block_id in range(len(self.conv2_x)):
            skip_con = self.skip_con2_x[block_id](tensor)
            if blocks_remain[blocks_count] != 0:
                for layer in self.conv2_x[block_id]:
                    tensor = layer(tensor)
                tensor *= blocks_remain[blocks_count]
            else:
                tensor = 0

            tensor = skip_con + tensor
            tensor = nn.functional.relu(tensor, inplace=True)

            blocks_count += 1

        for block_id in range(len(self.conv3_x)):
            skip_con = self.skip_con3_x[block_id](tensor)
            if blocks_remain[blocks_count] != 0:
                for layer in self.conv3_x[block_id]:
                    tensor = layer(tensor)
                tensor *= blocks_remain[blocks_count]
            else:
                tensor = 0

            tensor = skip_con + tensor
            tensor = nn.functional.relu(tensor, inplace=True)

            blocks_count += 1


        for block_id in range(len(self.conv4_x)):
            skip_con = self.skip_con4_x[block_id](tensor)
            if blocks_remain[blocks_count] != 0:
                for layer in self.conv4_x[block_id]:
                    tensor = layer(tensor)
                tensor *= blocks_remain[blocks_count]
            else:
                tensor = 0

            tensor = skip_con + tensor
            tensor = nn.functional.relu(tensor, inplace=True)

            blocks_count += 1


        for block_id in range(len(self.conv5_x)):
            skip_con = self.skip_con5_x[block_id](tensor)
            if blocks_remain[blocks_count] != 0:
                for layer in self.conv5_x[block_id]:
                    tensor = layer(tensor)
                tensor *= blocks_remain[blocks_count]
            else:
                tensor = 0

            tensor = skip_con + tensor
            tensor = nn.functional.relu(tensor, inplace=True)

            blocks_count += 1

        tensor = torch.mean(tensor, dim=(2, 3))
        return self.fcn(tensor)


    def setBlocksProbabilities(self, blocks_num, p_last=0.5):
        self.block_probs = []
        for block_id in range(blocks_num):
            prob = 1 - block_id / (blocks_num - 1) * (1 - p_last)
            self.block_probs.append(np.round(prob, 3))

    def backward(self, logits, labels, optimizer, print_loss=False, lb_epsi=0.1):
        loss = nn.CrossEntropyLoss(label_smoothing=lb_epsi)(logits, labels)
        preds = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        if print_loss:
            print(f"Loss: {loss.item()}, Accuracy: {accuracy}", end=" ")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()