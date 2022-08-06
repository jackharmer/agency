from typing import Tuple, Union
import torch
from dataclasses import dataclass
from agency.memory.block_memory import AgentStep, BlockOfSteps

from agency.memory.episodic import Roll


@dataclass
class PpoBatch:
    obs: torch.Tensor
    obs_next: torch.Tensor
    actions: torch.Tensor
    terminal_mask: torch.Tensor
    rewards: torch.Tensor
    policy_log_prob: torch.Tensor
    value: torch.Tensor
    num_rolls: int
    batch_size: int


class PpoBlockOfSteps(BlockOfSteps):
    def __init__(
        self,
        num_agents: int,
        num_elements: int,
        obs_size: Union[Tuple[int], int],
        num_actions: int,
        device: torch.device,
    ):
        self._num_agents = num_agents
        self._num_elements = num_elements
        block_size = (num_agents, num_elements)

        self._action_size = (num_actions,)
        self._value_size = (1,)
        self._obs_size = obs_size if type(obs_size) is tuple else (obs_size,)

        self.obs = torch.zeros(block_size + self._obs_size, device=device)
        self.obs_next = torch.zeros(block_size + self._obs_size, device=device)
        self.value = torch.zeros(block_size + self._value_size, device=device)
        self.reward = torch.zeros(block_size, device=device)
        self.action = torch.zeros(block_size + self._action_size, device=device)
        self.policy_log_prob = torch.zeros(block_size, device=device)
        self.done = torch.zeros(block_size, device=device)

    def write_at_index(self, step: AgentStep, write_index: int):
        self.obs[:, write_index, :] = step.obs
        self.obs_next[:, write_index, :] = step.obs_next
        self.value[:, write_index, :] = step.aux_data.value
        self.reward[:, write_index] = step.reward
        self.action[:, write_index, :] = step.policy.sample
        self.policy_log_prob[:, write_index] = step.policy.log_prob
        self.done[:, write_index] = step.done

    def sample_batch_at_indices(self, roll_idx_2d: torch.Tensor, agent_idx: torch.Tensor):
        num_rolls = roll_idx_2d.shape[0]
        roll_length = roll_idx_2d.shape[1]
        batch_size = num_rolls * roll_length

        obs = self._multidim_gather(self.obs, self._obs_size, roll_idx_2d, agent_idx)
        obs_next = self._multidim_gather(self.obs_next, self._obs_size, roll_idx_2d, agent_idx)
        obs_next = obs_next.reshape(num_rolls, roll_length, *self._obs_size)[:, -1, :]
        action = self._multidim_gather(self.action, self._action_size, roll_idx_2d, agent_idx)
        value = self._multidim_gather(self.value, self._value_size, roll_idx_2d, agent_idx)

        reward = torch.gather(self.reward[agent_idx], 1, roll_idx_2d)
        done = torch.gather(self.done[agent_idx], 1, roll_idx_2d)
        policy_log_prob = torch.gather(self.policy_log_prob[agent_idx], 1, roll_idx_2d)

        return PpoBatch(
            obs=obs.detach(),
            obs_next=obs_next.view(num_rolls, *self._obs_size).detach(),
            actions=action.detach(),
            terminal_mask=(1.0 - done).detach(),  # [numRolls,1]
            rewards=reward.detach(),  # [numRolls, rollLength]
            policy_log_prob=policy_log_prob.view(-1).detach(),
            value=value.detach(),
            num_rolls=num_rolls,
            batch_size=batch_size,
        )


@torch.jit.script
def nested_stack_jit(list_of_list_of_tensors: list[list[torch.Tensor]]):
    return torch.stack([torch.stack(x) for x in list_of_list_of_tensors])


def create_batch(rolls: list[Roll], device: torch.device):
    num_rolls = len(rolls)
    batch_size = sum([r.count() for r in rolls])

    obs = torch.stack([obs for r in rolls for obs in r.obs])
    obs_next = torch.stack([r.obs_next for r in rolls])

    actions = torch.cat([policy.sample for r in rolls for policy in r.policies])
    policy_log_probs = torch.cat([policy.log_prob for r in rolls for policy in r.policies])
    values = torch.cat([aux_data.value for r in rolls for aux_data in r.aux_data])

    dones = nested_stack_jit([r.dones for r in rolls])
    term_masks = 1.0 - dones
    rewards = nested_stack_jit([r.rewards for r in rolls])

    batch = PpoBatch(
        obs=obs.to(device),
        obs_next=obs_next.to(device),
        actions=actions.to(device),
        terminal_mask=term_masks.to(device),  # [numRolls,1]
        rewards=rewards.to(device),  # [numRolls, rollLength]
        policy_log_prob=policy_log_probs.to(device),
        value=values.to(device),
        num_rolls=num_rolls,
        batch_size=batch_size,
    )
    return batch


def create_batch_from_block_memory(batch: PpoBatch, device: torch.device):
    batch.obs.to(device)
    batch.obs_next.to(device)
    batch.actions.to(device)
    batch.terminal_mask.to(device)
    batch.rewards.to(device)
    batch.policy_log_prob.to(device)
    batch.value.to(device)
    return batch
