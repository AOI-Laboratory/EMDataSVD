# Author Information
# Name: Eugene Su
# Email: su.eugene@gmail.com
# GitHub: https://github.com/EugenePig

import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    def __init__(self):
        # Initialize the DUVModel class and its submodules
        super(DUVModel, self).__init__()

        # Define the ResNet feature extractor and the OCD encoder
        self.resnet_e = FeatureNet18()
        self.enc_ocd = Encoder_OCD()

    def forward(self, Ex_Us, Ex_V, Ey_Us, Ey_V, Ez_Us, Ez_V):
        # Perform matrix multiplications on each input pair (U, V) and expand dimensions
        Ex = torch.unsqueeze(torch.einsum('ijk,ikl->ijl', Ex_Us, Ex_V), dim=1)
        Ey = torch.unsqueeze(torch.einsum('ijk,ikl->ijl', Ey_Us, Ey_V), dim=1)
        Ez = torch.unsqueeze(torch.einsum('ijk,ikl->ijl', Ez_Us, Ez_V), dim=1)

        # Concatenate Ex, Ey, and Ez along the channel dimension
        x = torch.cat((Ex, Ey, Ez), dim=1)

        # Pass the concatenated tensor through the ResNet feature extractor
        x = self.resnet_e(x)

        # Reshape the extracted features to a 2D format (batch_size, channels, flattened spatial dimensions)
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        # Pass reshaped features through the OCD encoder
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
        # Initialize the Encoder_Feature_Linear class with two linear layers and an activation
        super(Encoder_Feature_Linear, self).__init__()

        # Define a two-layer linear transformation with an optional activation
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)
        self.activation = activation

    def forward(self, x):
        # Pass the input through the first linear layer and apply activation
        x = self.linear1(x)
        x = self.activation(x)

        # Pass through the second linear layer and apply activation
        x = self.linear2(x)
        x = self.activation(x)

        # Expand dimensions of the output tensor for compatibility with other modules
        return torch.unsqueeze(x, dim=1)


class Encoder_OCD(nn.Module):
    def __init__(self, inplanes=196, activation=nn.ReLU()):
        # Initialize the Encoder_OCD class with layers and submodules
        super(Encoder_OCD, self).__init__()

        # Set output size and define dimensions for intermediate layers
        out_size = 4
        dims = [out_size * inplanes, 512, 256, 6]

        # Save input planes and create a list of linear layers for feature encoding
        self.inplanes = inplanes
        self.feature_linears = nn.ModuleList([Encoder_Feature_Linear(in_size=512, out_size=out_size) for _ in range(inplanes)])

        # Define layers for flattening and transforming feature dimensions
        self.flatten = nn.Flatten()

        # Define the OCD transformation with three linear layers
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        # Set activation functions
        self.activation = activation
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        # Encode each feature channel individually using the corresponding linear layer
        features = list()
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        # Concatenate all encoded feature outputs along the channel dimension
        x = torch.cat(features, dim=1)

        # Flatten the concatenated features to feed into the linear layers
        x = self.flatten(x)

        # Pass through the first OCD linear layer and apply activation
        x = self.ocd_linear1(x)
        x = self.activation(x)

        # Pass through the second OCD linear layer and apply activation
        x = self.ocd_linear2(x)
        x = self.activation(x)

        # Pass through the final OCD linear layer with ReLU6 activation
        x = self.ocd_linear3(x)
        x = self.relu6(x)

        return x
