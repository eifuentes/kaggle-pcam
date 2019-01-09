from collections import OrderedDict

import torch
import torch.nn as nn


def BatchNorm2d(in_channels, init_weight=1, *args, **kwargs):
    """ Batch Normalization with Explicit Weight Initialization

        Weights initialized to 1 for ResNet fn to start as
        an identify operation. Faster initial training.
    """
    if not isinstance(init_weight, (int, float)):
        raise ValueError('init_weight must be an int or float')
    init_weight = float(init_weight)
    module = nn.BatchNorm2d(in_channels, *args, **kwargs)
    with torch.no_grad():
        module.weight.fill_(init_weight)
        module.bias.zero_()
    return module


def Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
           bias=False, init_fn=nn.init.kaiming_normal_):
    """ Conv2d Layer with Explicit Weight Initialization
        & Common Arg Assignments
    """
    module = nn.Conv2d(in_channels, out_channels,
                       kernel_size, stride, padding, bias=bias)
    with torch.no_grad():
        init_fn(module.weight, mode='fan_out')
        if hasattr(module, 'bias') and hasattr(module.bias, 'data'):
            module.bias.zero_()
    return module


class BasicBlock(nn.Module):
    """ Wide Residual Network Basic Block """
    def __init__(self, in_channels, out_channels, stride,
                 dropout_proba=0.0, residual_scale_factor=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_proba = dropout_proba

        # residuals scaling parameter (see Inception-v4, Szegedy et al.)
        if not (residual_scale_factor <= 0.3 or residual_scale_factor >= 0.1):
            raise ValueError('residual_scale_factor must be between 0.1 & 0.3')
        self.residual_scale_factor = residual_scale_factor

        # initial conv layer
        self.bn0 = nn.BatchNorm2d(self.in_channels)
        self.relu0 = nn.ReLU(inplace=True)
        # add skip connection if there is a change in the number of channels
        if self.in_channels != self.out_channels:
            self.skip0 = Conv2d(self.in_channels, self.out_channels,
                                kernel_size=1, stride=self.stride, padding=0)
        else:
            self.skip0 = lambda x: x
        self.conv0 = Conv2d(self.in_channels, self.out_channels, kernel_size=3,
                            stride=self.stride, padding=1)

        # optional dropout layer
        if self.dropout_proba > 0.0:
            self.dropout0 = nn.Dropout(self.dropout_proba, inplace=True)
        else:
            self.dropout0 = lambda x: x

        # last conv layer in block
        self.bn1 = BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(self.out_channels, self.out_channels,
                            kernel_size=3, stride=1, padding=1)

    def layer0(self, x):
        x = self.bn0(x)
        x = self.relu0(x)
        # placement of resnet skip connections follows preact ResNet pattern
        r = self.skip0(x)
        x = self.conv0(x)
        return x, r

    def layer1(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        return x

    def forward(self, x):
        x, r = self.layer0(x)
        x = self.dropout0(x)
        x = self.layer1(x) * self.residual_scale_factor
        return x.add_(r)


def calculate_channels(init_num_channels, num_groups, channel_scale_factor):
    num_channels = [init_num_channels]
    for group_num in range(num_groups):
        num_channels.append(
            init_num_channels*(2**group_num) * channel_scale_factor)
    return num_channels


def Group(num_blocks, in_channels, out_channels, stride,
          dropout_proba=0.0, residual_scale_factor=0.2):
    blocks = [
        (
            'block0',
            BasicBlock(in_channels, out_channels, stride, dropout_proba)
        )
    ]
    for block_num in range(1, num_blocks):
        blocks.append(
            (
                f'block{block_num}',
                BasicBlock(out_channels, out_channels,
                           stride=1,
                           dropout_proba=dropout_proba,
                           residual_scale_factor=residual_scale_factor)
            )
        )
    return nn.Sequential(OrderedDict(blocks))


class WideResNet(nn.Module):
    def __init__(self, in_channels, num_groups, num_blocks_per_group,
                 channel_scale_factor=1, init_num_channels=16,
                 dropout_proba=0.0, residual_scale_factor=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.num_blocks_per_group = num_blocks_per_group
        self.channel_scale_factor = channel_scale_factor

        # calculate seq. channel expansion with width channel scale factor
        self.num_channels = calculate_channels(init_num_channels,
                                               self.num_groups,
                                               self.channel_scale_factor)
        self.num_features = self.num_channels[-1]

        # initial convolution layer to expand image space channels
        layers = [
            (
                'conv0',
                Conv2d(self.in_channels, init_num_channels,
                       kernel_size=3, stride=1, padding=1)
            )
        ]

        # create groups
        for group_num in range(self.num_groups):
            stride = 1 if group_num == 0 else 2
            group = Group(num_blocks=self.num_blocks_per_group,
                          in_channels=self.num_channels[group_num],
                          out_channels=self.num_channels[group_num + 1],
                          stride=stride,
                          dropout_proba=dropout_proba,
                          residual_scale_factor=residual_scale_factor)
            layers.append(
                (f'group{group_num}', group)
            )
        # add final batch norm, activation, and adaptive avg pooling
        layers += [
            ('bnfin', nn.BatchNorm2d(self.num_features)),
            ('relufin', nn.ReLU(inplace=True)),
            ('adapoolfin', nn.AdaptiveAvgPool2d(1))
        ]
        # end to end convolutional feature extractor
        self.features = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class WideResNetBinaryClassifier(nn.Module):
    def __init__(self, in_channels,
                 num_groups, num_blocks_per_group,
                 channel_scale_factor=1, init_num_channels=16,
                 dropout_proba=0.0, residual_scale_factor=0.2):
        super().__init__()

        self.features = WideResNet(in_channels, num_groups,
                                   num_blocks_per_group,
                                   channel_scale_factor, init_num_channels,
                                   dropout_proba, residual_scale_factor)
        self.classifier = nn.Linear(self.features.num_features, 1, bias=True)

        # initialize linear layer weights and bias
        nn.init.normal_(self.classifier.weight, std=1e-3)
        nn.init.constant_(self.classifier.bias, 0.)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)
