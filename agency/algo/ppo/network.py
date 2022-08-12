from dataclasses import dataclass
from typing import Any, Union
import torch

import torch.nn as nn
from agency.layers.distributions import (
    CategoricalPolicy,
    DiscretePolicy,
    GaussianPolicy,
    DiscreteDistParams,
    ContinuousDistParams,
)
from agency.layers.feed_forward import InputNormalizer, conv_encoder, mlp
from agency.layers.policy_heads import PolicyHead
from agency.layers.value_heads import VHead
from agency.tools.helpers import calculate_conv_output_size


@dataclass
class PpoNetwork:
    value: Any
    policy: Any

    def to(self, device):
        self.value.to(device)
        self.policy.to(device)
        return self


@dataclass
class PpoParams:
    p_learning_rate: float = 0.0004
    v_learning_rate: float = 0.0004
    ppo_clip: float = 0.2
    v_clip: float = 0.2
    clip_value_function: bool = False
    entropy_loss_scaling: float = 0.001
    normalize_advantage: bool = False
    use_dual_optimizer: bool = False  # Use different optimizers for the value function and policy nets
    use_terminal_masking: bool = True
    use_state_independent_std: bool = True


@dataclass
class MlpNetworkArchitecture:
    shared_hidden_sizes: list[int]
    v_hidden_sizes: list[int]
    p_hidden_sizes: list[int]


@dataclass
class ConvNetworkArchitecture:
    shared_enc_channels: list[int]
    shared_enc_kernel_sizes: list[int]
    shared_enc_strides: list[int]
    v_hidden_sizes: list[int]
    p_hidden_sizes: list[int]


def create_distribution_layer(dist, input_size: int, num_actions: int, algo: PpoParams):
    if type(dist) is ContinuousDistParams:
        policy = GaussianPolicy(
            input_size, num_actions, use_state_independent_std=algo.use_state_independent_std
        )
    elif type(dist) is DiscreteDistParams:
        if dist.categorical:
            policy = CategoricalPolicy(input_size, num_actions)
        else:
            policy = DiscretePolicy(input_size, num_actions, dist.temperature, dist.hard)
    else:
        raise Exception("Unknown distribution type")
    return policy


def create_network(
    input_size: int,
    arch: MlpNetworkArchitecture,
    dist: Union[ContinuousDistParams, DiscreteDistParams],
    algo: PpoParams,
    num_actions: int,
):
    if arch.shared_hidden_sizes is None:
        print("using seperate network architectures")
        assert arch.v_hidden_sizes is not None
        assert arch.p_hidden_sizes is not None
        enc_input_size = input_size
    else:
        print("using shared network architecture")
        enc_input_size = arch.shared_hidden_sizes[-1]

    v_head_input_size = (
        arch.shared_hidden_sizes[-1] if arch.v_hidden_sizes is None else arch.v_hidden_sizes[-1]
    )
    p_head_input_size = (
        arch.shared_hidden_sizes[-1] if arch.p_hidden_sizes is None else arch.p_hidden_sizes[-1]
    )

    # create the value and policy encoders.
    v_enc = mlp(enc_input_size, arch.v_hidden_sizes) if arch.v_hidden_sizes is not None else nn.Identity()
    p_enc = mlp(enc_input_size, arch.p_hidden_sizes) if arch.p_hidden_sizes is not None else nn.Identity()

    # preprend a shared encoder, if required.
    if arch.shared_hidden_sizes is not None:
        shared_enc = mlp(input_size, arch.shared_hidden_sizes)
        v_enc = nn.Sequential(shared_enc, v_enc)
        p_enc = nn.Sequential(shared_enc, p_enc)

    return PpoNetwork(
        value=VHead(v_enc, input_size=v_head_input_size),
        policy=PolicyHead(
            p_enc,
            create_distribution_layer(dist, p_head_input_size, num_actions, algo),
        ),
    )


def create_vision_network(
    input_size: int,
    arch: ConvNetworkArchitecture,
    dist: Union[ContinuousDistParams, DiscreteDistParams],
    algo: PpoParams,
    num_actions: int,
    normalize_input: bool = False,
):
    enc_input_size = calculate_conv_output_size(
        input_size[1:3],
        arch.shared_enc_channels[-1],
        arch.shared_enc_kernel_sizes,
        arch.shared_enc_strides,
    )

    v_head_input_size = (
        arch.shared_hidden_sizes[-1] if arch.v_hidden_sizes is None else arch.v_hidden_sizes[-1]
    )
    p_head_input_size = (
        arch.shared_hidden_sizes[-1] if arch.p_hidden_sizes is None else arch.p_hidden_sizes[-1]
    )

    # create the value and policy encoders.
    v_enc = mlp(enc_input_size, arch.v_hidden_sizes) if arch.v_hidden_sizes is not None else nn.Identity()
    p_enc = mlp(enc_input_size, arch.p_hidden_sizes) if arch.p_hidden_sizes is not None else nn.Identity()

    shared_enc = conv_encoder(
        input_channels=input_size[0],
        layer_channels=arch.shared_enc_channels,
        layer_kernels_sizes=arch.shared_enc_kernel_sizes,
        layer_strides=arch.shared_enc_strides,
        flatten=True,
    )
    v_enc = nn.Sequential(shared_enc, v_enc)
    p_enc = nn.Sequential(shared_enc, p_enc)
    if normalize_input:
        normalizer = InputNormalizer(input_size)
        v_enc = nn.Sequential(normalizer, v_enc)
        p_enc = nn.Sequential(normalizer, p_enc)

    return PpoNetwork(
        value=VHead(v_enc, input_size=v_head_input_size),
        policy=PolicyHead(
            p_enc,
            create_distribution_layer(dist, p_head_input_size, num_actions, algo),
        ),
    )
