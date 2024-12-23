# Author Information
# Name: Eugene Su
# Email: su.eugene@gmail.com
# GitHub: https://github.com/EugenePig

import torch
from torch import nn
from ResNet import FeatureNet18


class PreProcess(nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()

        # Define a series of convolution layers with ReLU activation
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Pass input through the first convolution layer and apply activation
        x = self.conv1(x)
        x = self.activation(x)

        # Pass through the second convolution layer and apply activation
        x = self.conv2(x)
        x = self.activation(x)

        # Final convolution layer followed by activation
        x = self.conv3(x)
        x = self.activation(x)

        return x


class DUVModel(nn.Module):
    def __init__(self):
        super(DUVModel, self).__init__()

        # Pre-process layers for Ex, Ey, Ez input channels
        self.ex_preprocess = PreProcess()
        self.ey_preprocess = PreProcess()
        self.ez_preprocess = PreProcess()

        # Initialize main feature extraction and encoding layers
        self.resnet_e = FeatureNet18()  # Feature extraction network (ResNet)
        self.enc_ocd = Encoder_OCD()  # Encoder for OCD (Output Coding Dimension)

    def forward(self, Ex, Ey, Ez):
        # Apply preprocessing on each of the input components (Ex, Ey, Ez)
        Ex = self.ex_preprocess(Ex)
        Ey = self.ey_preprocess(Ey)
        Ez = self.ez_preprocess(Ez)

        # Concatenate the processed Ex, Ey, and Ez tensors along the channel dimension
        x = torch.cat((Ex, Ey, Ez), dim=1)

        # Pass through the feature extraction network
        x = self.resnet_e(x)

        # Reshape the output for the encoder
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        # Pass through the OCD encoder to generate the final output
        ocd = self.enc_ocd(x)

        return ocd


def conv3x3_2d(in_planes, out_planes, padding=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=padding, bias=bias)


def conv1x1_2d(in_planes, out_planes, bias=True):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv1x1_1d(in_planes, out_planes, padding=0, bias=True):
    """1x1 convolution without padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=padding, bias=bias)


def conv3x3_1d(in_planes, out_planes, padding=1, bias=True):
    """1x1 convolution without padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=padding, bias=bias)

def conv5x5_1d(in_planes, out_planes, padding=2, bias=True):
    """1x1 convolution without padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=1,
                     padding=padding, bias=bias)


class Encoder_Feature_Linear(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU()):
        super(Encoder_Feature_Linear, self).__init__()

        # Define two fully connected layers with specified input and output sizes
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)
        self.activation = activation

    def forward(self, x):
        # Pass input through the first linear layer and activation
        x = self.linear1(x)
        x = self.activation(x)

        # Pass through the second linear layer and activation
        x = self.linear2(x)
        x = self.activation(x)

        # Add an extra dimension for compatibility in further processing
        return torch.unsqueeze(x, dim=1)


class Encoder_OCD(nn.Module):
    def __init__(self, inplanes=196, activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()

        # Define output size and intermediate layer dimensions
        out_size = 8
        dims = [out_size * inplanes, 512, 256, 6]

        self.inplanes = inplanes

        # Define a list of linear layers for feature extraction for each inplane
        self.feature_linears = nn.ModuleList(
            [Encoder_Feature_Linear(in_size=512, out_size=out_size) for _ in range(inplanes)])

        # Define layers for OCD encoding
        self.flatten = nn.Flatten()
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        self.activation = activation
        self.relu6 = nn.ReLU6()  # ReLU activation limited to a max output of 6

    def forward(self, x):
        # Process each input plane separately through its corresponding feature linear layer
        features = list()
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        # Concatenate all feature outputs along the channel dimension
        x = torch.cat(features, dim=1)

        # Flatten the concatenated output to prepare for linear layers
        x = self.flatten(x)

        # Pass through three linear layers with activation functions
        x = self.ocd_linear1(x)
        x = self.activation(x)

        x = self.ocd_linear2(x)
        x = self.activation(x)

        # Final linear layer followed by ReLU6 activation
        x = self.ocd_linear3(x)
        x = self.relu6(x)

        return x
