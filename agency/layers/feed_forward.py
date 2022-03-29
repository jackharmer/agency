import math
import numpy as np
import torch
import torch.nn as nn


# ortho_init is FROM OPENAI BASELINES
# https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/tf_layers.py
def custom_ortho_init_(input_weights, gain=1.0):
    shape = tuple(input_weights.shape)
    if len(shape) == 2:
        flat_shape = shape
    # elif len(shape) == 4:  # assumes NHWC
    #     flat_shape = (np.prod(shape[:-1]), shape[-1])
    elif len(shape) == 4:  # assumes NHWC NCHW
        flat_shape = (np.prod([shape[0], shape[2], shape[3]]), shape[1])
    else:
        raise NotImplementedError
    gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
    weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
    weights = weights.reshape(shape)
    weights = (gain * weights[:shape[0], :shape[1]]).astype(np.float32)
    input_weights = weights


def mlp(
    input_size,
    layer_sizes,
):
    layer_sizes = [input_size] + layer_sizes

    layers = []
    for cc in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[cc], layer_sizes[cc + 1])
        custom_ortho_init_(linear.weight, gain=math.sqrt(2.0))
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def conv_encoder(
    input_channels,
    layer_channels,
    layer_kernels_sizes,
    layer_strides,
    flatten=False
):
    num_layers = len(layer_channels)

    layer_channels_full = [input_channels] + layer_channels

    layers = []
    for cc in range(num_layers):
        conv = nn.Conv2d(
            in_channels=layer_channels_full[cc],
            out_channels=layer_channels_full[cc + 1],
            kernel_size=layer_kernels_sizes[cc],
            stride=layer_strides[cc],
            padding=0
        )
        custom_ortho_init_(conv.weight, gain=math.sqrt(2.0))
        nn.init.constant_(conv.bias, 0.0)

        layers.append(conv)
        layers.append(nn.ReLU())

    if flatten:
        layers.append(torch.nn.Flatten())

    return nn.Sequential(*layers)


class GradientBlocker(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, state):
        with torch.no_grad():
            x = self._layer(state)
        return x
