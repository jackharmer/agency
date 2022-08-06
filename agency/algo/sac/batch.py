from typing import Any, Tuple, Union
import torch
from dataclasses import dataclass
import numpy as np
from agency.memory.block_memory import AgentStep, BlockOfSteps

from agency.memory.episodic import Roll


@dataclass
class SacBatch:
    obs: torch.Tensor
    obs_next: torch.Tensor
    actions: torch.Tensor
    terminal_mask: torch.Tensor
    rewards: torch.Tensor
    num_rolls: int
    batch_size: int


@dataclass
class SacBatchCategorical:
    obs: torch.Tensor
    obs_next: torch.Tensor
    sample: torch.Tensor
    terminal_mask: torch.Tensor
    rewards: torch.Tensor
    num_rolls: int
    batch_size: int


class SacBlockOfSteps(BlockOfSteps):
    def __init__(
        self,
        num_agents: int,
        num_elements: int,
        obs_size: Union[Tuple[int], int],
        num_actions: int,
        device: torch.device,
        categorical: bool = False,
    ):
        self._num_agents = num_agents
        self._num_elements = num_elements
        self._categorical = categorical
        block_size = (num_agents, num_elements)

        self._action_size = (num_actions,)
        self._obs_size = obs_size if type(obs_size) is tuple else (obs_size,)

        self.obs = torch.zeros(block_size + self._obs_size, device=device)
        self.obs_next = torch.zeros(block_size + self._obs_size, device=device)

        self.action = torch.zeros(block_size + self._action_size, device=device)

        self.done = torch.zeros(block_size, device=device)
        self.reward = torch.zeros(block_size, device=device)

    def write_at_index(self, step: AgentStep, write_index: int):
        self.obs[:, write_index, :] = step.obs
        self.obs_next[:, write_index, :] = step.obs_next
        self.action[:, write_index, :] = step.policy.sample
        self.reward[:, write_index] = step.reward
        self.done[:, write_index] = step.done

    def sample_batch_at_indices(self, roll_idx_2d: torch.Tensor, agent_idx: torch.Tensor):
        num_rolls = roll_idx_2d.shape[0]
        roll_length = roll_idx_2d.shape[1]
        batch_size = num_rolls * roll_length

        obs = self._multidim_gather(self.obs, self._obs_size, roll_idx_2d, agent_idx)
        obs_next = self._multidim_gather(self.obs_next, self._obs_size, roll_idx_2d, agent_idx)
        obs_next = obs_next.reshape(num_rolls, roll_length, *self._obs_size)[:, -1, :]
        action = self._multidim_gather(self.action, self._action_size, roll_idx_2d, agent_idx)

        reward = torch.gather(self.reward[agent_idx], 1, roll_idx_2d)
        done = torch.gather(self.done[agent_idx], 1, roll_idx_2d)

        if self._categorical:
            return SacBatchCategorical(
                obs=obs.detach(),
                obs_next=obs_next.view(num_rolls, *self._obs_size).detach(),
                sample=action.detach(),
                terminal_mask=(1.0 - done).detach(),  # [numRolls,1]
                rewards=reward.detach(),  # [numRolls, rollLength]
                num_rolls=num_rolls,
                batch_size=batch_size,
            )
        else:
            return SacBatch(
                obs=obs.detach(),
                obs_next=obs_next.view(num_rolls, *self._obs_size).detach(),
                actions=action.detach(),
                terminal_mask=(1.0 - done).detach(),  # [numRolls,1]
                rewards=reward.detach(),  # [numRolls, rollLength]
                num_rolls=num_rolls,
                batch_size=batch_size,
            )


def create_batch_from_block_memory(batch: SacBatch, device: torch.device):
    batch.obs.to(device)
    batch.obs_next.to(device)
    batch.actions.to(device)
    batch.terminal_mask.to(device)
    batch.rewards.to(device)
    return batch


def create_categorical_batch_from_block_memory(batch: SacBatchCategorical, device: torch.device):
    batch.obs.to(device)
    batch.obs_next.to(device)
    batch.sample.to(device)
    batch.terminal_mask.to(device)
    batch.rewards.to(device)
    return batch


@torch.jit.script
def nested_stack_jit(list_of_list_of_tensors: list[list[torch.Tensor]]):
    return torch.stack([torch.stack(x) for x in list_of_list_of_tensors])


def create_batch(rolls: list[Roll], device: Any):
    num_rolls = len(rolls)
    batch_size = sum([r.count() for r in rolls])

    obs = torch.stack([obs for r in rolls for obs in r.obs])
    obs_next = torch.stack([r.obs_next for r in rolls])
    actions = torch.cat([policy.sample for r in rolls for policy in r.policies])

    dones = nested_stack_jit([r.dones for r in rolls])
    term_masks = 1.0 - dones
    rewards = nested_stack_jit([r.rewards for r in rolls])

    return SacBatch(
        obs=obs.to(device),
        obs_next=obs_next.to(device),
        actions=actions.to(device),
        terminal_mask=term_masks.to(device),  # [numRolls,1]
        rewards=rewards.to(device),  # [numRolls, rollLength]
        num_rolls=num_rolls,
        batch_size=batch_size,
    )


def create_batch_categorical(rolls: list[Roll], device: Any):
    num_rolls = len(rolls)
    batch_size = sum([r.count() for r in rolls])

    obs = torch.stack([obs for r in rolls for obs in r.obs])
    obs_next = torch.stack([r.obs_next for r in rolls])
    # prob = torch.cat([policy.probs for r in rolls for policy in r.policies])
    sample = torch.cat([policy.sample for r in rolls for policy in r.policies])

    dones = nested_stack_jit([r.dones for r in rolls])
    term_masks = 1.0 - dones
    rewards = nested_stack_jit([r.rewards for r in rolls])

    return SacBatchCategorical(
        obs=obs.to(device),
        obs_next=obs_next.to(device),
        sample=sample.to(device),
        terminal_mask=term_masks.to(device),  # [numRolls,1]
        rewards=rewards.to(device),  # [numRolls, rollLength]
        num_rolls=num_rolls,
        batch_size=batch_size,
    )
