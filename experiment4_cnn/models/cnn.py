"""Small VGG-style CNN for 28x28 MNIST images, scaled by a width c.

    [conv3x3(1->c) BN ReLU] [conv3x3(c->c) BN ReLU] maxpool2          # 28 -> 14
    [conv3x3(c->2c) BN ReLU] [conv3x3(2c->2c) BN ReLU] maxpool2       # 14 -> 7
    global average pool -> [dropout, only if dropout > 0] -> Linear(2c -> 10)

Convolutions use padding=1 and no bias (the BN that follows has one). The data pipeline gives
flattened 784-value images, so the model reshapes them to 1x28x28 itself.
"""
import torch.nn as nn


def conv_bn_relu(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


def build_model(width=32, dropout=0.0):
    c = width
    return nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),
        *conv_bn_relu(1, c), *conv_bn_relu(c, c), nn.MaxPool2d(2),
        *conv_bn_relu(c, 2 * c), *conv_bn_relu(2 * c, 2 * c), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        *([nn.Dropout(dropout)] if dropout > 0 else []),
        nn.Linear(2 * c, 10),
    )
