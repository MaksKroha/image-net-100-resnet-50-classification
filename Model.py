import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # conv1_x
        self.main_conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3)
        self.main_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1, 0, 0))

        # realized bottleneck blocks in "full pre-activation" style
        self.conv2_x = []
        input_size = 64
        blocks_quality = 3
        for i in range(blocks_quality):
            self.conv2_x.append([])
            self.conv2_x[i].append(nn.BatchNorm2d(input_size))
            self.conv2_x[i].append(nn.Conv2d(input_size, 64, kernel_size=1, bias=False))

            self.conv2_x[i].append(nn.BatchNorm2d(64))
            self.conv2_x[i].append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))

            self.conv2_x[i].append(nn.BatchNorm2d(64))
            self.conv2_x[i].append(nn.Conv2d(64, 256, kernel_size=1, bias=False))
            input_size = 256


        self.conv3_x = []
        blocks_quality = 4
        for i in range(blocks_quality):
            self.conv3_x.append([])
            self.conv3_x[i].append(nn.BatchNorm2d(input_size))
            self.conv3_x[i].append(nn.Conv2d(input_size, 128, kernel_size=1, bias=False))

            self.conv3_x[i].append(nn.BatchNorm2d(128))
            self.conv3_x[i].append(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False))

            self.conv3_x[i].append(nn.BatchNorm2d(128))
            self.conv3_x[i].append(nn.Conv2d(128, 512, kernel_size=1, bias=False))
            input_size = 512


        self.conv4_x = []
        blocks_quality = 6
        for i in range(blocks_quality):
            self.conv4_x.append([])
            self.conv4_x[i].append(nn.BatchNorm2d(input_size))
            self.conv4_x[i].append(nn.Conv2d(input_size, 256, kernel_size=1, bias=False))

            self.conv4_x[i].append(nn.BatchNorm2d(256))
            self.conv4_x[i].append(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))

            self.conv4_x[i].append(nn.BatchNorm2d(256))
            self.conv4_x[i].append(nn.Conv2d(256, 1024, kernel_size=1, bias=False))
            input_size = 1024


        self.conv5_x = []
        blocks_quality = 3
        for i in range(blocks_quality):
            self.conv5_x.append([])
            self.conv5_x[i].append(nn.BatchNorm2d(input_size))
            self.conv5_x[i].append(nn.Conv2d(input_size, 512, kernel_size=1, bias=False))

            self.conv5_x[i].append(nn.BatchNorm2d(512))
            self.conv5_x[i].append(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False))

            self.conv5_x[i].append(nn.BatchNorm2d(512))
            self.conv5_x[i].append(nn.Conv2d(512, 2048, kernel_size=1, bias=False))
            input_size = 2048

        # Global average pooling
        self.fcn = nn.Linear(2048, 1000)







