from dataclasses import dataclass
from typing import Any

import torch
from agency.algo.sac.network import (ContinuousDistParams, DiscreteDistParams,
                                     MlpNetworkArchitecture,
                                     VisionNetworkArchitecture)
from agency.layers.distributions import DiscretePolicy, GaussianPolicy
from agency.layers.feed_forward import (mlp, conv_encoder, GradientBlocker)
from agency.layers.policy_heads import PolicyHead
from agency.layers.value_heads import (QEncoder, QHead,
                                       QVisionEncoder, soft_update)
from agency.tools.helpers import calculate_conv_output_size


@dataclass
class AwacParams:
    beta: float = 0.2
    adv_clip: float = 2.0
    use_softmax: bool = False


@dataclass
class AwacData:
    q1: Any
    q2: Any
    q1_target: Any
    q2_target: Any
    policy: Any
    policy_params: Any

    def __post_init__(self):
        # update q target nets to have the same weights as q networks
        self.update_target_weights(tau=1.0)

    def update_target_weights(self, tau=0.005):
        # update q target nets to have the same weights as q networks
        soft_update(self.q1_target, self.q1, tau=tau)
        soft_update(self.q2_target, self.q2, tau=tau)

    def to(self, device):
        self.q1.to(device)
        self.q2.to(device)
        self.q1_target.to(device)
        self.q2_target.to(device)
        self.policy.to(device)
        return self


def create_continuous_network(
        input_size: int,
        arch: MlpNetworkArchitecture,
        dist: ContinuousDistParams,
        algo: AwacParams,
        num_actions: int,
):
    q_input_size = input_size + num_actions

    q1_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q2_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q1_target_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q2_target_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)

    policy_encoder = mlp(input_size=input_size, layer_sizes=arch.p_hidden_sizes)
    policy_dist = GaussianPolicy(arch.p_hidden_sizes[-1], num_actions)
    policy = PolicyHead(policy_encoder, policy_dist)

    return AwacData(
        q1=QHead(QEncoder(q1_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q2=QHead(QEncoder(q2_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q1_target=QHead(QEncoder(q1_target_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q2_target=QHead(QEncoder(q2_target_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        policy=policy,
        policy_params=list(policy.parameters())
    )


def create_discrete_network(
        input_size: int,
        arch: MlpNetworkArchitecture,
        dist: DiscreteDistParams,
        algo: AwacParams,
        num_actions: int,
):
    q_input_size = input_size + num_actions

    policy_encoder = mlp(input_size=input_size, layer_sizes=arch.p_hidden_sizes)
    q1_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q2_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q1_target_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)
    q2_target_encoder = mlp(input_size=q_input_size, layer_sizes=arch.q_hidden_sizes)

    policy_dist = DiscretePolicy(arch.p_hidden_sizes[-1], num_actions, temperature=dist.temperature)
    policy = PolicyHead(policy_encoder, policy_dist)

    return AwacData(
        q1=QHead(QEncoder(q1_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q2=QHead(QEncoder(q2_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q1_target=QHead(QEncoder(q1_target_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        q2_target=QHead(QEncoder(q2_target_encoder), input_size=arch.q_hidden_sizes[-1], output_size=1),
        policy=policy,
        policy_params=list(policy.parameters())
    )


def create_discrete_vision_network(
        input_size: list[int],
        arch: VisionNetworkArchitecture,
        dist: DiscreteDistParams,
        algo: AwacParams,
        num_actions: int,
):
    shared_enc_output_size = calculate_conv_output_size(
        input_size[0:2],
        arch.shared_enc_channels[-1],
        arch.shared_enc_kernel_sizes,
        arch.shared_enc_strides
    )
    q_input_size = shared_enc_output_size + num_actions
    q_output_size = 1

    def make_conv_encoder():
        return torch.nn.Sequential(
            conv_encoder(
                input_channels=input_size[2],
                layer_channels=arch.shared_enc_channels,
                layer_kernels_sizes=arch.shared_enc_kernel_sizes,
                layer_strides=arch.shared_enc_strides,
            ),
            torch.nn.Flatten()
        )

    def make_q_encoder(_shared_encoder):
        return QVisionEncoder(
            _shared_encoder,
            mlp(
                input_size=q_input_size,
                layer_sizes=arch.q_hidden_sizes,
            )
        )

    if arch.use_shared_encoder:
        shared_encoder = make_conv_encoder()
        shared_target_encoder = make_conv_encoder()
        policy_body = mlp(
            input_size=shared_enc_output_size,
            layer_sizes=arch.p_hidden_sizes,
        )

        policy_encoder = torch.nn.Sequential(
            GradientBlocker(shared_encoder),
            policy_body
        )

        q1_encoder = make_q_encoder(shared_encoder)
        q2_encoder = make_q_encoder(shared_encoder)
        # TODO: Shared_target_encoder will get updated twice when copying the weights.
        # This shouldn't cause much of a problem though.
        q1_target_encoder = make_q_encoder(shared_target_encoder)
        q2_target_encoder = make_q_encoder(shared_target_encoder)
    else:
        policy_encoder = torch.nn.Sequential(
            make_conv_encoder(),
            mlp(
                input_size=shared_enc_output_size,
                layer_sizes=arch.p_hidden_sizes,
            )
        )
        q1_encoder = make_q_encoder(make_conv_encoder())
        q2_encoder = make_q_encoder(make_conv_encoder())
        q1_target_encoder = make_q_encoder(make_conv_encoder())
        q2_target_encoder = make_q_encoder(make_conv_encoder())

    policy_dist = DiscretePolicy(arch.p_hidden_sizes[-1], num_actions, temperature=dist.temperature)

    policy = PolicyHead(policy_encoder, policy_dist)

    if arch.use_shared_encoder:
        policy_params = list(policy_body.parameters()) + list(policy_dist.parameters())
    else:
        policy_params = list(policy.parameters())

    net = AwacData(
        q1=QHead(q1_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q2=QHead(q2_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q1_target=QHead(q1_target_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q2_target=QHead(q2_target_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        policy=policy,
        policy_params=policy_params
    )
    return net


def create_vision_network(
        input_size: list[int],
        arch: VisionNetworkArchitecture,
        dist: ContinuousDistParams,
        algo: AwacParams,
        num_actions: int,
):
    shared_enc_output_size = calculate_conv_output_size(
        input_size[0:2],
        arch.shared_enc_channels[-1],
        arch.shared_enc_kernel_sizes,
        arch.shared_enc_strides
    )
    q_input_size = shared_enc_output_size + num_actions
    q_output_size = 1

    def make_conv_encoder():
        return torch.nn.Sequential(
            conv_encoder(
                input_channels=input_size[2],
                layer_channels=arch.shared_enc_channels,
                layer_kernels_sizes=arch.shared_enc_kernel_sizes,
                layer_strides=arch.shared_enc_strides,
            ),
            torch.nn.Flatten()
        )

    def make_q_encoder(_shared_encoder):
        return QVisionEncoder(
            _shared_encoder,
            mlp(
                input_size=q_input_size,
                layer_sizes=arch.q_hidden_sizes,
            )
        )

    if arch.use_shared_encoder:
        shared_encoder = make_conv_encoder()
        shared_target_encoder = make_conv_encoder()
        policy_body = mlp(
            input_size=shared_enc_output_size,
            layer_sizes=arch.p_hidden_sizes,
        )

        policy_encoder = torch.nn.Sequential(
            GradientBlocker(shared_encoder),
            policy_body
        )

        q1_encoder = make_q_encoder(shared_encoder)
        q2_encoder = make_q_encoder(shared_encoder)
        # TODO: Shared_target_encoder will get updated twice when copying the weights.
        # This shouldn't cause much of a problem though.
        q1_target_encoder = make_q_encoder(shared_target_encoder)
        q2_target_encoder = make_q_encoder(shared_target_encoder)
    else:
        policy_encoder = torch.nn.Sequential(
            make_conv_encoder(),
            mlp(
                input_size=shared_enc_output_size,
                layer_sizes=arch.p_hidden_sizes,
            )
        )
        q1_encoder = make_q_encoder(make_conv_encoder())
        q2_encoder = make_q_encoder(make_conv_encoder())
        q1_target_encoder = make_q_encoder(make_conv_encoder())
        q2_target_encoder = make_q_encoder(make_conv_encoder())

    policy_dist = GaussianPolicy(arch.p_hidden_sizes[-1], num_actions)
    policy = PolicyHead(policy_encoder, policy_dist)

    if arch.use_shared_encoder:
        policy_params = list(policy_body.parameters()) + list(policy_dist.parameters())
    else:
        policy_params = list(policy.parameters())

    net = AwacData(
        q1=QHead(q1_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q2=QHead(q2_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q1_target=QHead(q1_target_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        q2_target=QHead(q2_target_encoder, input_size=arch.q_hidden_sizes[-1], output_size=q_output_size),
        policy=policy,
        policy_params=policy_params
    )
    return net