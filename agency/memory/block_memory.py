import collections
import pickle
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple

import pgzip
import torch


@dataclass
class AgentStep:
    obs: torch.Tensor
    obs_next: torch.Tensor
    reward: torch.Tensor
    policy: Any
    done: torch.Tensor
    aux_data: Any
    agent_id: list[int]


@dataclass
class BlockOfSteps(ABC):
    @abstractmethod
    def write_at_index(self, step: AgentStep, write_index: int):
        pass

    @abstractmethod
    def sample_batch_at_indices(self, roll_idx_2d: torch.Tensor, agent_idx: torch.Tensor):
        pass

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def num_elements(self):
        return self._num_elements

    @torch.jit.export
    def _get_idx(self, roll_idx_2d: torch.Tensor, feature_size: Tuple[int]):
        return roll_idx_2d.reshape(roll_idx_2d.shape + (1,) * len(feature_size)).repeat((1, 1) + feature_size)

    @torch.jit.export
    def _multidim_gather(
        self,
        feature: torch.Tensor,
        feature_size: Tuple[int],
        roll_idx_2d: torch.Tensor,
        agent_idx: torch.Tensor,
    ):
        flat01_feature = feature.reshape(self._num_agents * self._num_elements, *feature_size)
        flat01_roll_idx = roll_idx_2d + self._num_elements * agent_idx.view(-1, 1)
        roll_idx = self._get_idx(flat01_roll_idx, feature_size)
        idx = roll_idx.reshape(roll_idx_2d.shape[0] * roll_idx_2d.shape[1], *feature_size)
        return torch.gather(flat01_feature, 0, idx)


class BlockOfStepsMemory:
    def __init__(self, block: BlockOfSteps, device: torch.device):
        self._device = device
        self._block = block
        self._max_size = block.num_elements
        self._num_agents = block.num_agents

        self._lock = threading.RLock()
        self.clear()

    # @torch.jit.export
    def append(self, step: AgentStep, **kwargs):
        with self._lock:
            self._block.write_at_index(step, self._write_index)
            self._completed_episodes_counter += int(step.done.sum())
            self._agent_step_counter += self._num_agents
            self._write_index += 1
            self._have_valid_data_until_index += 1
            self._write_index = self._write_index % self._max_size
            self._have_valid_data_until_index = min(self._have_valid_data_until_index, self._max_size)

    def num_columns(self) -> int:
        with self._lock:
            return self._have_valid_data_until_index

    def num_completed_episodes(self) -> int:
        with self._lock:
            return self._completed_episodes_counter

    def num_completed_steps(self) -> int:
        with self._lock:
            return self._agent_step_counter

    def num_stored_steps(self) -> int:
        with self._lock:
            return self._num_agents * self._have_valid_data_until_index

    def capacity(self) -> int:
        with self._lock:
            return self._num_agents * self._max_size

    def is_full(self) -> bool:
        with self._lock:
            return self._have_valid_data_until_index == (self._max_size - 1)

    def clear(self):
        self._write_index = 0
        self._have_valid_data_until_index = 0
        self._agent_step_counter = 0
        self._completed_episodes_counter = 0

    @torch.jit.export
    def sample(self, batch_size: int, roll_length: int):
        with self._lock:
            num_rolls = batch_size // roll_length

            # sample a random agent for each rollout
            agent_idx = torch.randint(0, self._num_agents, (num_rolls,), device=self._device)

            # For each picked agent now sample a rollout
            if self._have_valid_data_until_index != self._max_size:
                # The memory hasn't yet been filled, thus some data is invalid. Limit sampling to valid data range.
                roll_range_start = 0
                roll_range_end = self._have_valid_data_until_index - roll_length + 1
            else:
                # If all the memory is valid, then start sampling after the current write index, and never cross
                # the write index.
                roll_range_start = self._write_index
                roll_range_end = roll_range_start + self._max_size - roll_length + 1

            roll_start_idx = (
                torch.randint(roll_range_start, roll_range_end, (num_rolls, 1), device=self._device)
                % self._max_size
            )

            roll_idx_2d = (
                (torch.arange(0, roll_length, device=self._device).repeat(num_rolls, 1) + roll_start_idx)
                % self._max_size
            ).long()

            # Now sample the rollout data and return.
            return self._block.sample_batch_at_indices(roll_idx_2d, agent_idx)

    # TODO: Add unit test
    def save(self, fname, compressed=False):
        print(f"Saving memory to: {fname}")
        save_data = {
            "write_index": self._write_index,
            "have_valid_data_until_index": self._have_valid_data_until_index,
            "agent_step_counter": self._agent_step_counter,
            "completed_episodes_counter": self._completed_episodes_counter,
            "block": self._block,
        }
        if compressed:
            with pgzip.open(fname, "wb") as f:
                pickle.dump(save_data, f)
        else:
            with open(fname, "wb") as f:
                pickle.dump(save_data, f)
        print(f"-- saved")

    # TODO: Add unit test
    def load(self, fname, compressed=False):
        print(f"Loading memory from: {fname}")
        if compressed:
            with pgzip.open(fname, "rb") as f:
                load_data = pickle.load(f)
        else:
            with open(fname, "rb") as f:
                load_data = pickle.load(f)
        self._write_index = load_data["write_index"]
        self._have_valid_data_until_index = load_data["have_valid_data_until_index"]
        self._agent_step_counter = load_data["agent_step_counter"]
        self._completed_episodes_counter = load_data["completed_episodes_counter"]
        self._block = load_data["block"]
        print(f"-- loaded {self.num_stored_steps()} steps")
