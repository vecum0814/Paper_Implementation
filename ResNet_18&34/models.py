import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                            nn.BatchNorm2d(out_channels)
                            )

        else:
            self.downsample = None

    def forward(self, x):
        i = x
        out = self.conv_block(x)

        if self.downsample is not None:
            i = self.downsample(i)

        out += i
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.base = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.gap = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def modeltype(model):
    if model == 'resnet18':
        return ResNet(ResidualBlock, [2, 2, 2, 2])

    elif model == 'resnet34':
        return ResNet(ResidualBlock, [3, 4, 6, 3])

