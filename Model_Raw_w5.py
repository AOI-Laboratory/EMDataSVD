import torch
from torch import nn
from ResNet import FeatureNet18


class PreProcess(nn.Module):
    def __init__(self):
        super(PreProcess, self).__init__()

        # Define three convolutional layers with ReLU activation
        self.conv1 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Pass input through the first convolutional layer and apply ReLU
        x = self.conv1(x)
        x = self.activation(x)

        # Pass through the second convolutional layer and apply ReLU
        x = self.conv2(x)
        x = self.activation(x)

        # Pass through the third convolutional layer and apply ReLU
        x = self.conv3(x)
        x = self.activation(x)

        return x


class DUVModel(nn.Module):
    def __init__(self):
        super(DUVModel, self).__init__()

        # Initialize separate preprocessing networks for Ex, Ey, and Ez inputs
        self.ex_preprocess = PreProcess()
        self.ey_preprocess = PreProcess()
        self.ez_preprocess = PreProcess()

        # Define feature extraction network and OCD encoder
        self.resnet_e = FeatureNet18()
        self.enc_ocd = Encoder_OCD()

    def forward(self, Ex, Ey, Ez):
        # Apply preprocessing to Ex, Ey, Ez inputs
        Ex = self.ex_preprocess(Ex)
        Ey = self.ey_preprocess(Ey)
        Ez = self.ez_preprocess(Ez)

        # Concatenate processed inputs along the channel dimension
        x = torch.cat((Ex, Ey, Ez), dim=1)

        # Pass concatenated inputs through feature extraction network
        x = self.resnet_e(x)

        # Reshape output for OCD encoding
        x = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))

        # Apply OCD encoding
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

        # Define two linear layers with an activation function in between
        self.linear1 = nn.Linear(in_size, out_size * 4, bias=True)
        self.linear2 = nn.Linear(out_size * 4, out_size, bias=True)
        self.activation = activation

    def forward(self, x):
        # Pass input through first linear layer and apply activation
        x = self.linear1(x)
        x = self.activation(x)

        # Pass through second linear layer and apply activation
        x = self.linear2(x)
        x = self.activation(x)

        # Return output with an added dimension
        return torch.unsqueeze(x, dim=1)


class Encoder_OCD(nn.Module):
    def __init__(self, inplanes=196, activation=nn.ReLU()):
        super(Encoder_OCD, self).__init__()
        out_size = 8
        dims = [out_size * inplanes, 512, 256, 6]

        self.inplanes = inplanes

        # Create a list of linear feature encoders for each input plane
        self.feature_linears = nn.ModuleList(
            [Encoder_Feature_Linear(in_size=512, out_size=out_size) for _ in range(inplanes)])

        # Flatten layer for flattening the features
        self.flatten = nn.Flatten()

        # Define three linear layers for OCD processing
        self.ocd_linear1 = nn.Linear(dims[0], dims[1], bias=True)
        self.ocd_linear2 = nn.Linear(dims[1], dims[2], bias=True)
        self.ocd_linear3 = nn.Linear(dims[2], dims[3], bias=False)

        # Set activation functions
        self.activation = activation
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        features = list()

        # Pass each plane through its respective feature encoder
        for i in range(self.inplanes):
            features.append(self.feature_linears[i](x[:, :, i]))

        # Concatenate features along the second dimension
        x = torch.cat(features, dim=1)

        # Flatten the concatenated features
        x = self.flatten(x)

        # Apply OCD linear layers with activations in between
        x = self.ocd_linear1(x)
        x = self.activation(x)

        x = self.ocd_linear2(x)
        x = self.activation(x)

        x = self.ocd_linear3(x)
        x = self.relu6(x)

        return x