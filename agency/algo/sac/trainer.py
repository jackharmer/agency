from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from agency.core.batch import Batch, CategoricalBatch
from agency.core.inference import Inferer
from agency.core.logger import TrainLogInfo, log_grads_and_vars
from agency.tools.gamma_matrix import discount, make_gamma_matrix
from agency.tools.helpers import clip_grad_norm_, tensor_to_numpy


@dataclass
class TrainData:
    policy_optimizer: Any
    q1_optimizer: Any
    q2_optimizer: Any
    alpha_optimizer: Any
    gamma_matrix: torch.Tensor
    target_entropy: torch.Tensor

    clip_norm: float
    reward_scaling: float
    reward_clip_value: float
    train_alpha: bool


def train_on_batch(net: Any, td: TrainData, mb: Batch) -> TrainLogInfo:
    debug_grads = False
    scalar_logs, dist_logs, image_logs = {}, {}, {}

    rewards = torch.clamp(mb.rewards * td.reward_scaling, -td.reward_clip_value, td.reward_clip_value)
    alpha = torch.exp(net.log_alpha).detach()

    # policy update
    curr_policy = net.policy(mb.obs)
    E_Q = net.q2(mb.obs, curr_policy.sample).squeeze()
    soft_v = E_Q + alpha * (-curr_policy.log_prob)
    p_loss = -soft_v.mean()

    # policy backprop
    td.policy_optimizer.zero_grad()
    p_loss.backward()
    if debug_grads:
        log_grads_and_vars("policy", net.policy.named_parameters(), scalar_logs, dist_logs)
    p_grad_norm = clip_grad_norm_(net.policy_params, td.clip_norm)
    td.policy_optimizer.step()

    # soft target
    with torch.no_grad():
        curr_policy_obs_next = net.policy(mb.obs_next)
        q_i_next = torch.minimum(
            net.q1_target(mb.obs_next, curr_policy_obs_next.sample),
            net.q2_target(mb.obs_next, curr_policy_obs_next.sample)
        )
        curr_policy_next_entropy = -curr_policy_obs_next.log_prob
        soft_v_next = q_i_next + alpha * curr_policy_next_entropy.unsqueeze(1)
        soft_v_next_masked = soft_v_next * mb.terminal_mask
        soft_v_bootstrap = discount(rewards, soft_v_next_masked, td.gamma_matrix).detach()

    # q1 update
    q1_pred = net.q1(mb.obs, mb.actions)
    error1 = (q1_pred - soft_v_bootstrap)
    q1_loss = 0.5 * (error1**2).mean()

    # q1 backprop
    td.q1_optimizer.zero_grad()
    q1_loss.backward()
    if debug_grads:
        log_grads_and_vars("q1", net.q1.named_parameters(), scalar_logs, dist_logs)
    q1_grad_norm = clip_grad_norm_(net.q1.parameters(), td.clip_norm)
    td.q1_optimizer.step()

    # q2 update
    q2_pred = net.q2(mb.obs, mb.actions)
    error2 = (q2_pred - soft_v_bootstrap)
    q2_loss = 0.5 * (error2**2).mean()

    # q2 backprop
    td.q2_optimizer.zero_grad()
    q2_loss.backward()
    if debug_grads:
        log_grads_and_vars("q2", net.q2.named_parameters(), scalar_logs, dist_logs)
    q2_grad_norm = clip_grad_norm_(net.q2.parameters(), td.clip_norm)
    td.q2_optimizer.step()

    # alpha
    if td.train_alpha:
        policy = net.policy(mb.obs)
        target_entropy_error = (-policy.log_prob - td.target_entropy).detach()
        td.alpha_optimizer.zero_grad()
        alpha_loss = (net.log_alpha * target_entropy_error).mean()
        alpha_loss.backward()
        alpha_grad_norm = clip_grad_norm_(net.alpha_parameters(), td.clip_norm)
        td.alpha_optimizer.step()
    else:
        alpha_loss = torch.tensor(0.0)
        alpha_grad_norm = torch.tensor(0.0)
        target_entropy_error = torch.tensor(0.0)

    net.update_target_weights()

    # LOGGING INFO
    scalar_logs["obs/obs_mean"] = mb.obs.mean()
    # scalar_logs["obs/obs_next_sum"] = mb.obs_next.mean()
    scalar_logs["environment/terminal_ratio"] = (mb.terminal_mask.shape[0]-mb.terminal_mask.sum())/mb.terminal_mask.shape[0]
    # Grads
    scalar_logs["grads/policy_norm"] = tensor_to_numpy(p_grad_norm)
    scalar_logs["grads/q1_norm"] = tensor_to_numpy(q1_grad_norm)
    scalar_logs["grads/q2_norm"] = tensor_to_numpy(q2_grad_norm)
    scalar_logs["grads/alpha_norm"] = tensor_to_numpy(alpha_grad_norm)
    # Params
    scalar_logs["params/alpha"] = alpha
    scalar_logs["params/q_target"] = tensor_to_numpy(q_i_next.mean())
    scalar_logs["params/policy_entropy"] = tensor_to_numpy(-curr_policy_obs_next.log_prob.mean())
    scalar_logs["params/q1_pred"] = tensor_to_numpy(q1_pred.mean())
    scalar_logs["params/q2_pred"] = tensor_to_numpy(q2_pred.mean())
    scalar_logs["params/q1_error"] = tensor_to_numpy(error1.mean())
    scalar_logs["params/q2_error"] = tensor_to_numpy(error2.mean())
    scalar_logs["params/target_entropy_error"] = tensor_to_numpy(target_entropy_error.mean())
    scalar_logs["params/target_entropy"] = td.target_entropy
    # Losses
    scalar_logs["loss/q"] = tensor_to_numpy(q1_loss + q2_loss)
    scalar_logs["loss/policy"] = tensor_to_numpy(p_loss)
    scalar_logs["loss/alpha"] = tensor_to_numpy(alpha_loss)

    # image_logs["obs"] = tensor_to_numpy(mb.obs[0:3])
    # image_logs["next_obs"] = tensor_to_numpy(mb.obs_next[:3])

    return TrainLogInfo(
        samples=mb.batch_size,
        scalar_logs_dict=scalar_logs,
        dist_logs_dict=dist_logs,
        image_logs_dict=image_logs,
    )


def train_on_batch_categorical(net: Any, td: TrainData, mb: CategoricalBatch) -> TrainLogInfo:
    scalar_logs, dist_logs, image_logs = {}, {}, {}

    rewards = torch.clamp(mb.rewards * td.reward_scaling, -td.reward_clip_value, td.reward_clip_value)

    with torch.no_grad():
        alpha = torch.exp(net.log_alpha)

    clip = True

    # policy update
    with torch.no_grad():
        # Q = torch.minimum(net.q1(mb.obs), net.q2(mb.obs))
        Q = net.q1(mb.obs)
    curr_policy = net.policy(mb.obs)

    E_Q = (curr_policy.prob * Q).sum(dim=1)
    soft_v = E_Q + alpha * curr_policy.entropy
    p_loss = -soft_v.mean()

    # policy backprop
    td.policy_optimizer.zero_grad()
    p_loss.backward()
    p_grad_norm = clip_grad_norm_(net.policy_params, td.clip_norm, clip=clip)
    td.policy_optimizer.step()

    # soft target
    with torch.no_grad():
        curr_policy_next = net.policy(mb.obs_next)
        q_next = torch.minimum(net.q1_target(mb.obs_next), net.q2_target(mb.obs_next))

        v_next = (curr_policy_next.prob * q_next).sum(dim=1)
        soft_v_next = v_next + alpha * curr_policy_next.entropy

        soft_v_next_masked = soft_v_next.unsqueeze(1) * mb.terminal_mask
        soft_v_bootstrap = discount(rewards, soft_v_next_masked, td.gamma_matrix).squeeze()

    # q1 update
    q1_pred = (mb.sample * net.q1(mb.obs)).sum(dim=1)
    error1 = q1_pred - soft_v_bootstrap
    q1_loss = F.mse_loss(q1_pred, soft_v_bootstrap)

    # q1 backprop
    td.q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_grad_norm = clip_grad_norm_(net.q1.parameters(), td.clip_norm, clip=clip)
    td.q1_optimizer.step()

    # q2 update
    q2_pred = (mb.sample * net.q2(mb.obs)).sum(dim=1)
    error2 = q2_pred - soft_v_bootstrap
    q2_loss = F.mse_loss(q2_pred, soft_v_bootstrap)

    # q2 backprop
    td.q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_grad_norm = clip_grad_norm_(net.q2.parameters(), td.clip_norm, clip=clip)
    td.q2_optimizer.step()

    if td.train_alpha:
        # alpha
        policy = net.policy(mb.obs)
        with torch.no_grad():
            target_entropy_error = (policy.entropy - td.target_entropy)

        td.alpha_optimizer.zero_grad()
        alpha_loss = (net.log_alpha * target_entropy_error).mean()

        # alpha backprop
        alpha_loss.backward()
        alpha_grad_norm = clip_grad_norm_(net.alpha_parameters(), td.clip_norm, clip=clip)
        td.alpha_optimizer.step()
    else:
        alpha_loss = torch.FloatTensor(0.0)
        alpha_grad_norm = torch.FloatTensor(0.0)
        target_entropy_error = torch.FloatTensor(0.0)

    net.update_target_weights()

    # LOGGING INFO
    if len(mb.obs.shape) == 4:
        image_logs["obs/image"] = mb.obs[0:3, 0:3, :, :]
    scalar_logs["obs/obs_mean"] = mb.obs.mean()
    # scalar_logs["obs/obs_next_mean"] = mb.obs_next.mean()
    scalar_logs["environment/terminal_ratio"] = (mb.terminal_mask.shape[0] - mb.terminal_mask.sum())/mb.terminal_mask.shape[0]
    # Grads
    scalar_logs["grads/policy_norm"] = tensor_to_numpy(p_grad_norm)
    scalar_logs["grads/q1_norm"] = tensor_to_numpy(q1_grad_norm)
    scalar_logs["grads/q2_norm"] = tensor_to_numpy(q2_grad_norm)
    scalar_logs["grads/alpha_norm"] = tensor_to_numpy(alpha_grad_norm)
    # Params
    scalar_logs["params/alpha"] = alpha
    scalar_logs["params/v_target"] = tensor_to_numpy(v_next.mean())
    scalar_logs["params/q_target"] = tensor_to_numpy(q_next.mean())
    scalar_logs["params/policy_entropy"] = tensor_to_numpy(curr_policy_next.entropy.mean())
    scalar_logs["params/q1_pred"] = q1_pred.mean()
    scalar_logs["params/q2_pred"] = q2_pred.mean()
    scalar_logs["params/q1_error"] = error1.mean()
    scalar_logs["params/q2_error"] = error2.mean()
    scalar_logs["params/target_entropy_error"] = tensor_to_numpy(target_entropy_error.mean())
    scalar_logs["params/target_entropy"] = td.target_entropy
    # Losses
    scalar_logs["loss/q"] = tensor_to_numpy(q1_loss + q2_loss)
    scalar_logs["loss/policy"] = tensor_to_numpy(p_loss)
    scalar_logs["loss/alpha"] = tensor_to_numpy(alpha_loss)

    return TrainLogInfo(
        samples=mb.batch_size,
        scalar_logs_dict=scalar_logs,
        dist_logs_dict=dist_logs,
        image_logs_dict=image_logs
    )


def create_inferer(net, wp, device):
    return Inferer(
        net,
        wp.num_actions,
        device,
    )


def create_train_state_data(net, hp, wp, categorical=False):
    eps = 1e-7
    policy_optimizer = torch.optim.Adam(net.policy_params, lr=hp.learning_rate, eps=eps)
    q1_optimizer = torch.optim.Adam(net.q1.parameters(), lr=hp.learning_rate, eps=eps)
    q2_optimizer = torch.optim.Adam(net.q2.parameters(), lr=hp.learning_rate, eps=eps)
    alpha_optimizer = torch.optim.Adam(net.alpha_parameters(), lr=hp.learning_rate, eps=eps)

    if categorical:
        target_entropy_factor = np.log(wp.num_actions)  # with target_entropy_constant 0.2
    else:
        target_entropy_factor = -float(wp.num_actions)  # with target_entropy_constant 1.0

    train_data = TrainData(
        policy_optimizer=policy_optimizer,
        q1_optimizer=q1_optimizer,
        q2_optimizer=q2_optimizer,
        alpha_optimizer=alpha_optimizer,
        gamma_matrix=make_gamma_matrix(hp.gamma, hp.roll_length).to(hp.device),
        target_entropy=torch.tensor(hp.algo.target_entropy_constant * target_entropy_factor, dtype=torch.float32, requires_grad=False),
        clip_norm=hp.clip_norm,
        reward_scaling=hp.reward_scaling,
        reward_clip_value=hp.reward_clip_value,
        train_alpha=hp.algo.train_alpha,
    )
    return train_data
