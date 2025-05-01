import torch
from torch import nn
from src.utils.logger import exception_logger


class Model(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        torch.set_default_dtype(torch.float32)
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
            if i == 0:
                self.skip_con2_x.append(nn.Conv2d(input_size, 256, kernel_size=1, bias=False))
            else:
                self.skip_con2_x.append(nn.Identity())

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
            if i == 0 :
                self.skip_con3_x.append(nn.Conv2d(input_size, 512, kernel_size=1, bias=False))
            else:
                self.skip_con3_x.append(nn.Identity())

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
            if i == 0:
                self.skip_con4_x.append(nn.Conv2d(input_size, 1024, kernel_size=1, bias=False))
            else:
                self.skip_con4_x.append(nn.Identity())

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
            if i == 0:
                self.skip_con5_x.append(nn.Conv2d(input_size, 2048, kernel_size=1, bias=False))
            else:
                self.skip_con5_x.append(nn.Identity())

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
        self.dropout = nn.Dropout(p=dropout)
        self.fcn = nn.Linear(2048, output_classes)

    @exception_logger
    def forward(self, input_tensors):
        convs_x = [self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]
        skip_cons_x = [self.skip_con2_x, self.skip_con3_x, self.skip_con4_x, self.skip_con5_x]

        tensor = self.main_conv(input_tensors)
        tensor = self.main_max_pool(tensor)

        for i in range(len(convs_x)):
            for block_id in range(len(convs_x[i])):
                skip_con = skip_cons_x[i][block_id](tensor)

                for layer in convs_x[i][block_id]:
                    tensor = layer(tensor)

                tensor = skip_con + tensor
                tensor = nn.functional.relu(tensor, inplace=True)

        tensor = torch.mean(tensor, dim=[2, 3])
        tensor = self.dropout(tensor)
        tensor = self.fcn(tensor)

        return tensor
