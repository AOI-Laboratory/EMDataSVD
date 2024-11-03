import torch
from torch import nn
from ResNet import FeatureNet18


class DUVModel(nn.Module):
    def __init__(self):
        super(DUVModel, self).__init__()

        # Initialize 1x1 convolution to expand input channels from 1 to 3
        self.conv1 = conv1x1_2d(1, 3)

        # Initialize ResNet feature extraction model
        self.resnet_e = FeatureNet18()

        # Initialize custom encoder module for OCD (Objective Condition Detection or similar)
        self.enc_ocd = Encoder_OCD()

    def forward(self, x):
        # Apply 1x1 convolution to input
        x = self.conv1(x)

        # Extract features using ResNet-based feature extractor
        x = self.resnet_e(x)

        # Reshape feature map to prepare for encoding
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        # Encode features using the OCD encoder module
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

        # First linear layer to expand input to a higher dimension
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)

        # Second linear layer to reduce to the desired output dimension
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)

        # Activation function (default: ReLU)
        self.activation = activation

    def forward(self, x):
        # Apply first linear layer followed by activation
        x = self.linear1(x)
        x = self.activation(x)

        # Apply second linear layer followed by activation
        x = self.linear2(x)
        x = self.activation(x)

        # Add a new dimension at dim=1 for compatibility with subsequent processing
        return torch.unsqueeze(x, dim=1)


class Encoder_OCD(nn.Module):
    def __init__(self, inplanes=84, activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()

        # Set output size for each feature encoder
        out_size = 11

        # Define dimensions for the three fully connected layers
        dims = [out_size * inplanes, 512, 256, 6]

        # Number of input feature planes
        self.inplanes = inplanes

        # Create list of linear encoders, one for each input plane
        self.feature_linears = nn.ModuleList(
            [Encoder_Feature_Linear(in_size=512, out_size=out_size) for _ in range(inplanes)]
        )

        # Flatten layer for feature concatenation
        self.flatten = nn.Flatten()

        # Define fully connected layers for further encoding
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        # Activation functions
        self.activation = activation
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        # Encode each feature plane individually and collect outputs
        features = list()
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        # Concatenate features along the dimension for processing
        x = torch.cat(features, dim=1)

        # Flatten concatenated features for fully connected layers
        x = self.flatten(x)

        # Apply first fully connected layer with activation
        x = self.ocd_linear1(x)
        x = self.activation(x)

        # Apply second fully connected layer with activation
        x = self.ocd_linear2(x)
        x = self.activation(x)

        # Apply final linear layer with ReLU6 activation for output
        x = self.ocd_linear3(x)
        x = self.relu6(x)

        return x