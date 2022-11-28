from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from agency.algo.ppo.network import PpoNetwork
from agency.algo.ppo.network import PpoParams
from agency.algo.ppo.batch import PpoBatch
from agency.algo.ppo.inference import PpoInferer
from agency.core.logger import TrainLogInfo, log_grads_and_vars
from agency.layers.distributions import GaussianPolicySample
from agency.layers.feed_forward import InputNormalizer
from agency.tools.gamma_matrix import discount, make_gamma_matrix, gae_discount
from agency.tools.helpers import clip_grad_norm_, tensor_to_numpy


@dataclass
class PpoTrainData:
    p_optimizer: Any
    v_optimizer: Any
    gamma_matrix: torch.Tensor
    value_normalizer: Any


def create_train_state_data(net: PpoNetwork, hp, wp):
    eps = 1e-7
    # eps = 1e-5
    if not hp.algo.use_dual_optimizer:
        p_optimizer = torch.optim.Adam(
            list(net.value.parameters()) + list(net.policy.parameters()),
            lr=hp.algo.p_learning_rate,
            eps=eps,
        )
        v_optimizer = None
    else:
        p_optimizer = torch.optim.Adam(net.policy.parameters(), lr=hp.algo.p_learning_rate, eps=eps)
        v_optimizer = torch.optim.Adam(net.value.parameters(), lr=hp.algo.v_learning_rate, eps=eps)

    train_data = PpoTrainData(
        p_optimizer=p_optimizer,
        v_optimizer=v_optimizer,
        gamma_matrix=make_gamma_matrix(hp.rl.gamma, hp.rl.roll_length).to(hp.device),
        value_normalizer=InputNormalizer((1,), device=hp.device, clamp=False),
    )
    return train_data


def create_inferer(net, wp, device):
    return PpoInferer(
        net,
        wp.num_actions,
        device,
    )


@torch.jit.export
def train_on_batch(
    hp: Any, net: PpoNetwork, td: PpoTrainData, mb: PpoBatch, fetch_log_data: bool
) -> TrainLogInfo:
    debug_grads = False
    scalar_logs, dist_logs, image_logs = {}, {}, {}

    reward_mean = mb.rewards.mean()
    reward_min = mb.rewards.min()
    reward_max = mb.rewards.max()

    rewards = torch.clamp(
        mb.rewards * hp.rl.reward_scaling, -hp.rl.reward_clip_value, hp.rl.reward_clip_value
    )

    old_values = mb.value

    # forward pass
    v_pred = net.value(mb.obs)
    with torch.no_grad():
        v_pred_next = net.value(mb.obs_next)
    curr_policy = net.policy(mb.obs)

    # target value
    with torch.no_grad():
        if hp.algo.normalize_value:
            v_pred_next_unnormed = td.value_normalizer.reverse_norm(v_pred_next)
            v_pred_unnormed = td.value_normalizer.reverse_norm(v_pred)
        else:
            v_pred_unnormed = v_pred
            v_pred_next_unnormed = v_pred_next

        if hp.algo.use_terminal_masking:
            next_term_mask = mb.terminal_mask
        else:
            next_term_mask = torch.ones_like(mb.terminal_mask)

        if hp.algo.use_gae:
            v_bootstrap = gae_discount(
                rewards,
                v_pred_unnormed,
                v_pred_next_unnormed,
                next_term_mask,
                hp.rl.gamma,
                lamb=hp.algo.gae_lambda,
            ).detach()
        else:
            v_bootstrap = discount(rewards, v_pred_next_unnormed, td.gamma_matrix, next_term_mask).detach()

        advantage = v_bootstrap - v_pred_unnormed

        if hp.algo.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage = advantage.detach().squeeze()

        if hp.algo.normalize_value:
            v_bootstrap = td.value_normalizer(v_bootstrap)

    # value loss
    v_error = v_bootstrap - v_pred
    v_loss = v_error**2
    if hp.algo.clip_value_function:
        v_pred_delta = v_pred - old_values
        v_pred_clip = old_values + v_pred_delta.clamp(-hp.algo.v_clip, hp.algo.v_clip)
        v_loss_clip = (v_bootstrap - v_pred_clip) ** 2
        v_loss = torch.max(v_loss, v_loss_clip)
    v_loss = v_loss.mean()
    v_loss = hp.algo.v_loss_scaling * v_loss

    if hp.algo.normalize_value:
        v_pred = v_pred_unnormed

    # ratio = prob(a) / old_prob(a) = exp(prob(a) - old_prob(a))
    ratio = torch.exp(net.policy.log_prob_of_sample(mb.actions, curr_policy) - mb.policy_log_prob)

    surr1 = advantage * ratio
    surr2 = advantage * ratio.clamp(1.0 - hp.algo.ppo_clip, 1.0 + hp.algo.ppo_clip)
    p_loss = -torch.min(surr1, surr2)
    p_loss = p_loss.mean()

    # entropy loss
    entropy_loss = -hp.algo.entropy_loss_scaling * curr_policy.entropy
    entropy_loss = entropy_loss.mean()

    # backprop
    if hp.algo.use_dual_optimizer:
        p_loss = p_loss + entropy_loss

        td.p_optimizer.zero_grad()
        p_loss.backward()
        p_grad_norm = clip_grad_norm_(net.policy.parameters(), hp.backprop.clip_norm)
        td.p_optimizer.step()

        td.v_optimizer.zero_grad()
        v_loss.backward()
        v_grad_norm = clip_grad_norm_(net.value.parameters(), hp.backprop.clip_norm)
        td.v_optimizer.step()
    else:
        total_loss = p_loss + entropy_loss + 0.5 * v_loss

        td.p_optimizer.zero_grad()
        total_loss.backward()
        params = list(net.value.parameters()) + list(net.policy.parameters())
        p_grad_norm = clip_grad_norm_(params, hp.backprop.clip_norm)
        td.p_optimizer.step()

    if fetch_log_data:
        # LOGGING INFO
        scalar_logs["obs/obs_mean"] = mb.obs.mean()
        scalar_logs["obs/obs_min"] = mb.obs.min()
        scalar_logs["obs/obs_max"] = mb.obs.max()
        scalar_logs["obs/reward_mean"] = reward_mean
        scalar_logs["obs/reward_min"] = reward_min
        scalar_logs["obs/reward_max"] = reward_max
        scalar_logs["environment/terminal_ratio"] = (
            mb.terminal_mask.shape[0] - mb.terminal_mask.sum(1).clamp_max(1).sum()
        ) / mb.terminal_mask.shape[0]

        # Grads
        # scalar_logs["grads/norm"] = tensor_to_numpy(grad_norm)
        scalar_logs["grads/policy_norm"] = tensor_to_numpy(p_grad_norm)
        if hp.algo.use_dual_optimizer:
            scalar_logs["grads/value_norm"] = tensor_to_numpy(v_grad_norm)

        # Params
        scalar_logs["params/policy_entropy"] = tensor_to_numpy(curr_policy.entropy.mean())
        if type(curr_policy) is GaussianPolicySample:
            scalar_logs["params/policy_std"] = tensor_to_numpy(curr_policy.std.mean())
        scalar_logs["params/v_pred"] = tensor_to_numpy(v_pred.mean())
        scalar_logs["params/v_error"] = tensor_to_numpy(v_error.mean())
        scalar_logs["params/ratio"] = tensor_to_numpy(ratio.mean())

        # Losses
        scalar_logs["loss/value"] = tensor_to_numpy(v_loss)
        scalar_logs["loss/policy"] = tensor_to_numpy(p_loss)
        scalar_logs["loss/entropy"] = tensor_to_numpy(entropy_loss)

    return TrainLogInfo(
        samples=mb.batch_size,
        scalar_logs_dict=scalar_logs,
        dist_logs_dict=dist_logs,
        image_logs_dict=image_logs,
    )
