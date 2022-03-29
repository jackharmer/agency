from typing import Any
import torch
from dataclasses import dataclass
import numpy as np

from agency.memory.episodic import Roll


@dataclass
class Batch:
    obs: torch.Tensor
    obs_next: torch.Tensor
    actions: torch.Tensor
    terminal_mask: torch.Tensor
    rewards: torch.Tensor
    num_rolls: int
    batch_size: int


@dataclass
class CategoricalBatch:
    obs: torch.Tensor
    obs_next: torch.Tensor
    prob: torch.Tensor
    sample: torch.Tensor
    terminal_mask: torch.Tensor
    rewards: torch.Tensor
    num_rolls: int
    batch_size: int


def create_batch(rolls: list[Roll], device: Any):
    obs = np.array([obs for r in rolls for obs in r.obs])
    obs_next = np.array([r.obs_next for r in rolls])
    # This is slow:
    actions = np.concatenate([policy.sample for r in rolls for policy in r.policies])
    term_masks = np.array([1.0 - float(r.dones[-1]) for r in rolls])
    rewards = np.array([r.rewards for r in rolls])

    return Batch(
        obs=torch.FloatTensor(obs).to(device),
        obs_next=torch.FloatTensor(obs_next).to(device),
        actions=torch.FloatTensor(actions).to(device),
        terminal_mask=torch.FloatTensor(term_masks).unsqueeze(1).to(device),  # [numRolls,1]
        rewards=torch.FloatTensor(rewards).to(device),  # [numRolls, rollLength]
        num_rolls=len(rolls),
        batch_size=sum([r.count() for r in rolls])
    )


def create_batch_categorical(rolls: list[Roll], device: Any):
    # IMPORTANT, converting to np.array() first is 100x faster than passing the list to torch.tensor!
    obs = np.array([obs for r in rolls for obs in r.obs])
    obs_next = np.array([r.obs_next for r in rolls])
    prob = np.concatenate([policy.prob for r in rolls for policy in r.policies])
    sample = np.concatenate([policy.sample for r in rolls for policy in r.policies])
    term_masks = np.array([1.0 - float(r.dones[-1]) for r in rolls])
    rewards = np.array([r.rewards for r in rolls])

    return CategoricalBatch(
        obs=torch.FloatTensor(obs).to(device),
        obs_next=torch.FloatTensor(obs_next).to(device),
        prob=torch.FloatTensor(prob).to(device),
        sample=torch.FloatTensor(sample).to(device),
        terminal_mask=torch.FloatTensor(term_masks).unsqueeze(1).to(device),  # [numRolls,1]
        rewards=torch.FloatTensor(rewards).to(device),  # [numRolls, rollLength]
        num_rolls=len(rolls),
        batch_size=sum([r.count() for r in rolls])
    )
