import math
from typing import Any
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
    weights = (gain * weights[: shape[0], : shape[1]]).astype(np.float32)
    input_weights = weights


def mlp(
    input_size,
    layer_sizes,
    activation_layer=nn.ReLU,
):
    layer_sizes = [input_size] + layer_sizes

    layers = []
    for cc in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[cc], layer_sizes[cc + 1])
        custom_ortho_init_(linear.weight, gain=math.sqrt(2.0))
        nn.init.constant_(linear.bias, 0.0)
        layers.append(linear)
        layers.append(activation_layer())

    return nn.Sequential(*layers)


def conv_encoder(
    input_channels: int,
    layer_channels: list[int],
    layer_kernels_sizes: list[int],
    layer_strides: list[int],
    layer_padding: list[int] = None,
    flatten: bool = False,
    activation_layer: Any = nn.ReLU,
    use_group_norm: bool = False,
):
    num_layers = len(layer_channels)

    layer_channels_full = [input_channels] + layer_channels
    if layer_padding is None:
        layer_padding = [0] * len(layer_channels_full)

    layers = []
    for i in range(num_layers):
        conv = nn.Conv2d(
            in_channels=layer_channels_full[i],
            out_channels=layer_channels_full[i + 1],
            kernel_size=layer_kernels_sizes[i],
            stride=layer_strides[i],
            padding=layer_padding[i],
        )
        custom_ortho_init_(conv.weight, gain=math.sqrt(2.0))
        nn.init.constant_(conv.bias, 0.0)

        layers.append(conv)
        if use_group_norm:
            layers.append(nn.GroupNorm(1, layer_channels_full[i + 1]))
        layers.append(activation_layer())

    if flatten:
        layers.append(torch.nn.Flatten())

    return nn.Sequential(*layers)


def upsize_conv_encoder(
    output_channels: int,
    layer_channels: list[int],
    layer_kernels_sizes: list[int],
    layer_padding: list[int],
    layer_strides: list[tuple[int]] = None,
    layer_output_sizes: list[tuple[int]] = None,
    activation_layer=nn.ReLU,
    use_final_layer_activation: bool = False,
):
    num_layers = len(layer_channels)
    layer_channels_full = layer_channels + [output_channels]
    if layer_strides is None:
        layer_strides = [1] * len(layer_channels_full)

    layers = []
    for i in range(num_layers):
        up = torch.nn.Upsample(size=layer_output_sizes[i], mode="nearest")
        conv = nn.Conv2d(
            in_channels=layer_channels_full[i],
            out_channels=layer_channels_full[i + 1],
            kernel_size=layer_kernels_sizes[i],
            stride=layer_strides[i],
            padding=layer_padding[i],
        )
        # custom_ortho_init_(conv.weight, gain=math.sqrt(2.0))
        # nn.init.constant_(conv.bias, 0.0)

        layers.append(up)
        layers.append(conv)
        layers.append(activation_layer())

    if not use_final_layer_activation:
        layers.pop()

    return nn.Sequential(*layers)


class GradientBlocker(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, state):
        with torch.no_grad():
            x = self._layer(state)
        return x


class InputNormalizer(torch.nn.Module):
    def __init__(self, obs_shape, device="cuda", clamp=True, clamp_val=4.0, mean_dims: tuple[int] = (0,)):
        super().__init__()
        self.clamp = clamp
        self.clamp_val = clamp_val
        self.mean_dims = mean_dims
        self.register_buffer("count", torch.ones((), dtype=torch.float64, device=device))
        self.register_buffer("running_mean", torch.zeros(obs_shape, dtype=torch.float64, device=device))
        self.register_buffer("running_var", torch.ones(obs_shape, dtype=torch.float64, device=device))

    @torch.jit.export
    def update_normalization(self, obs):
        batch_size = obs.shape[0]
        total_count = self.count + batch_size

        mean_diff = obs.mean(self.mean_dims) - self.running_mean
        weighted_mean_diff_sq = mean_diff**2 * self.count * batch_size / total_count / total_count
        weighted_running_var = self.running_var * self.count / total_count
        weighted_obs_var = obs.var(self.mean_dims) * batch_size / total_count

        self.running_mean = self.running_mean + mean_diff * batch_size / total_count
        self.running_var = weighted_running_var + weighted_obs_var + weighted_mean_diff_sq
        self.count = total_count

    @torch.jit.export
    def normalize(self, obs):
        if self.training:
            self.update_normalization(obs)
        std = torch.sqrt(self.running_var.float() + 1e-5)
        obs = (obs - self.running_mean.float()) / std
        if self.clamp:
            obs = torch.clamp(obs, min=-self.clamp_val, max=self.clamp_val)
        return obs

    @torch.jit.export
    def reverse_norm(self, obs):
        std = torch.sqrt(self.running_var.float() + 1e-5)
        return (obs * std) + self.running_mean.float()

    @torch.jit.export
    def forward(self, obs):
        return self.normalize(obs)
