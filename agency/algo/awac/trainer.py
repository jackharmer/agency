from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from agency.algo.sac.batch import SacBatch
from agency.algo.sac.inference import Inferer
from agency.core.logger import TrainLogInfo
from agency.tools.gamma_matrix import discount, make_gamma_matrix
from agency.tools.helpers import tensor_to_numpy


@dataclass
class AwacTrainData:
    policy_optimizer: Any
    q1_optimizer: Any
    q2_optimizer: Any
    gamma_matrix: torch.Tensor
    beta: torch.Tensor

    clip_norm: float
    reward_scaling: float
    reward_clip_value: float
    adv_clip: float
    use_softmax: bool


def train_on_batch(net: Any, td: AwacTrainData, mb: SacBatch, fetch_log_data: bool) -> TrainLogInfo:
    scalar_logs, dist_logs, image_logs = {}, {}, {}

    rewards = torch.clamp(mb.rewards * td.reward_scaling, -td.reward_clip_value, td.reward_clip_value)

    # policy update
    td.policy_optimizer.zero_grad()
    curr_policy = net.policy(mb.obs)

    # calculate advantage weighting
    with torch.no_grad():
        E_Q = torch.minimum(
            net.q1(mb.obs, curr_policy.sample),
            net.q2(mb.obs, curr_policy.sample),
        )
        Q_i = torch.minimum(
            net.q1(mb.obs, mb.actions),
            net.q2(mb.obs, mb.actions),
        )
        advantage = torch.clamp(Q_i - E_Q, -td.adv_clip, td.adv_clip)
        scaled_advantage = advantage / td.beta
        if td.use_softmax:
            # normalise over batch
            advantage_weighting = F.softmax(scaled_advantage, dim=0) * float(advantage.shape[0])
        else:
            advantage_weighting = torch.exp(scaled_advantage)

    log_prob_i = net.policy.log_prob_of_sample(mb.actions, curr_policy)
    policy_loss = log_prob_i * advantage_weighting.detach().squeeze()
    p_loss = -policy_loss.mean()

    # policy backprop
    p_loss.backward()
    p_grad_norm = torch.nn.utils.clip_grad_norm_(net.policy.parameters(), td.clip_norm)
    td.policy_optimizer.step()

    # target value
    with torch.no_grad():
        curr_policy_next = net.policy(mb.obs_next)
        q_i_next = torch.minimum(
            net.q1_target(mb.obs_next, curr_policy_next.sample),
            net.q2_target(mb.obs_next, curr_policy_next.sample),
        )
        v_bootstrap = discount(rewards, q_i_next, td.gamma_matrix, mb.terminal_mask).detach()

    # q1 update
    td.q1_optimizer.zero_grad()
    q1_pred = net.q1(mb.obs, mb.actions)
    q1_loss = F.mse_loss(q1_pred, v_bootstrap)

    # q1 backprop
    q1_loss.backward()
    q1_grad_norm = torch.nn.utils.clip_grad_norm_(net.q1.parameters(), td.clip_norm)
    td.q1_optimizer.step()

    # q2 update
    td.q2_optimizer.zero_grad()
    q2_pred = net.q2(mb.obs, mb.actions)
    q2_loss = F.mse_loss(q2_pred, v_bootstrap)

    # q2 backprop
    q2_loss.backward()
    q2_grad_norm = torch.nn.utils.clip_grad_norm_(net.q2.parameters(), td.clip_norm)
    td.q2_optimizer.step()

    net.update_target_weights()

    if fetch_log_data:
        # logging info
        scalar_logs["environment/terminal_ratio"] = (
            mb.terminal_mask.shape[0] - mb.terminal_mask.sum()
        ) / mb.terminal_mask.shape[0]
        scalar_logs["grads/policy_norm"] = tensor_to_numpy(p_grad_norm)
        scalar_logs["grads/q1_norm"] = tensor_to_numpy(q1_grad_norm)
        scalar_logs["grads/q2_norm"] = tensor_to_numpy(q2_grad_norm)
        scalar_logs["params/q_target"] = tensor_to_numpy(q_i_next.mean())
        scalar_logs["params/policy_entropy"] = tensor_to_numpy((-curr_policy_next.log_prob).mean())
        scalar_logs["params/q1_pred"] = tensor_to_numpy(q1_pred.mean())
        scalar_logs["params/q2_pred"] = tensor_to_numpy(q2_pred.mean())
        scalar_logs["loss/q"] = tensor_to_numpy((q1_loss + q2_loss))
        scalar_logs["loss/policy"] = tensor_to_numpy(p_loss)

    return TrainLogInfo(
        samples=mb.batch_size,
        scalar_logs_dict=scalar_logs,
        dist_logs_dict=dist_logs,
        image_logs_dict=image_logs,
    )


def create_inferer(net, wp, device):
    return Inferer(
        net,
        wp.num_actions,
        device,
    )


def create_train_state_data(net, hp, wp):
    train_data = AwacTrainData(
        policy_optimizer=torch.optim.Adam(net.policy.parameters(), lr=hp.algo.learning_rate),
        q1_optimizer=torch.optim.Adam(net.q1.parameters(), lr=hp.algo.learning_rate),
        q2_optimizer=torch.optim.Adam(net.q2.parameters(), lr=hp.algo.learning_rate),
        gamma_matrix=make_gamma_matrix(hp.rl.gamma, hp.rl.roll_length).to(hp.device),
        beta=torch.tensor(hp.algo.beta, dtype=torch.float32),
        clip_norm=hp.backprop.clip_norm,
        reward_scaling=hp.rl.reward_scaling,
        reward_clip_value=hp.rl.reward_clip_value,
        adv_clip=hp.algo.adv_clip,
        use_softmax=hp.algo.use_softmax,
    )
    return train_data
