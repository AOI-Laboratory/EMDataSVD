import torch
from torch import nn
from ResNet import FeatureNet18


# Define DUVModel, which serves as the main model for this architecture
class DUVModel(nn.Module):
    def __init__(self):
        super(DUVModel, self).__init__()

        # Initialize a ResNet-based feature extraction network and OCD encoder
        self.resnet_e = FeatureNet18()  # ResNet-18 for initial feature extraction
        self.enc_ocd = Encoder_OCD()  # Encoder for OCD processing

    def forward(self, x):
        # Pass input through the ResNet feature extractor
        x = self.resnet_e(x)

        # Reshape feature output to be compatible with OCD encoder
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        # Pass reshaped feature map into OCD encoder
        ocd = self.enc_ocd(x)

        return ocd  # Return output from OCD encoder


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


# Define a linear feature encoder with activation functions
class Encoder_Feature_Linear(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU()):
        super(Encoder_Feature_Linear, self).__init__()

        # Define two linear layers with bias, the first expands features
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)

        # Activation function (default: ReLU)
        self.activation = activation

    def forward(self, x):
        # Apply first linear transformation and activation
        x = self.linear1(x)
        x = self.activation(x)

        # Apply second linear transformation and activation
        x = self.linear2(x)
        x = self.activation(x)

        # Unsqueeze to add a dimension for compatibility
        return torch.unsqueeze(x, dim=1)


# Define the OCD encoder network
class Encoder_OCD(nn.Module):
    def __init__(self, inplanes=196, activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()

        # Configuration for output size and layer dimensions
        out_size = 8
        dims = [out_size * inplanes, 512, 256, 6]  # Layer dimensions

        # Number of feature planes
        self.inplanes = inplanes

        # Create a list of linear feature encoders for each plane
        self.feature_linears = nn.ModuleList([Encoder_Feature_Linear(in_size=512, out_size=out_size)
                                              for _ in range(inplanes)])

        # Flatten layer to prepare for dense layers
        self.flatten = nn.Flatten()

        # Define three linear layers for OCD processing
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        # Activation functions
        self.activation = activation
        self.relu6 = nn.ReLU6()  # ReLU6 limits output to [0, 6]

    def forward(self, x):
        # Encode features for each plane and append to a list
        features = list()
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        # Concatenate all encoded features along the dimension 1
        x = torch.cat(features, dim=1)

        # Flatten concatenated features for dense layer processing
        x = self.flatten(x)

        # Pass through three linear layers with activations
        x = self.ocd_linear1(x)
        x = self.activation(x)

        x = self.ocd_linear2(x)
        x = self.activation(x)

        # Final linear layer with ReLU6 activation to constrain output
        x = self.ocd_linear3(x)
        x = self.relu6(x)

        return x  # Return final encoded output