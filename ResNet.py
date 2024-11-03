import torch.nn as nn


class basic_block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, downsample):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FeatureNet18(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), net_block=basic_block):
        super(FeatureNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.net_block_layer(net_block, 64, layers[0])
        self.layer2 = self.net_block_layer(net_block, 128, layers[1], stride=2)
        self.layer3 = self.net_block_layer(net_block, 256, layers[2], stride=2)
        self.layer4 = self.net_block_layer(net_block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def net_block_layer(self, net_block, out_channels, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * net_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * net_block.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * net_block.expansion))

        layers = []
        layers.append(net_block(self.in_channels, out_channels, stride, downsample))
        if net_block.expansion != 1:
            self.in_channels = out_channels * net_block.expansion

        else:
            self.in_channels = out_channels

        for i in range(1, num_blocks):
            layers.append(net_block(self.in_channels, out_channels, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


