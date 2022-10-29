from collections import deque
import itertools
import numpy as np
import threading
from typing import Any
from dataclasses import dataclass

from agency.memory.block_memory import AgentStep


@dataclass
class EpisodicMemoryStep:
    obs: Any
    policy: Any
    reward: float
    done: bool
    obs_next: Any
    aux_data: Any


class Roll:
    def __init__(self, steps: list[EpisodicMemoryStep] = None, clone=False):
        self.obs: list[Any] = []
        self.obs_next: Any = None
        self.policies: list[Any] = []
        self.aux_data: list[Any] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        if steps is not None:
            self.append_steps(steps, clone=clone)

    def append_steps(self, steps: list[EpisodicMemoryStep], clone=False):
        for step in steps:
            self.append(step, clone)

    def append(self, step: EpisodicMemoryStep, clone=False):
        # TODO do .to(device) here?
        self.obs.append(step.obs)
        # obs_next contain the last obs in the rollout
        self.obs_next = step.obs_next
        self.policies.append(step.policy)
        self.aux_data.append(step.aux_data)
        self.rewards.append(step.reward)
        self.dones.append(step.done)

    def count(self):
        return len(self.obs)


class EpisodicMemory:
    def __init__(self, max_size: int, min_episode_length: int = 10, debug: bool = False):
        self._lock = threading.RLock()
        self._min_episode_length = min_episode_length  # This should generall be > rollout length
        self._max_total_steps = max_size
        self._agent_step_counter = 0
        self._debug = debug

        # self._completed_episodes = deque()
        self._completed_episodes = []
        self._active_episodes = {}
        self._active_episodes_that_can_be_sampled = {}

    def append(self, step: AgentStep, **kwargs):
        with self._lock:
            num_agents = len(step.agent_id)
            for cc in range(num_agents):
                ep_step = EpisodicMemoryStep(
                    obs=step.obs[cc],
                    policy=step.policy[cc],
                    reward=step.reward[cc],
                    done=step.done[cc].float(),
                    obs_next=step.obs_next[cc],
                    aux_data=step.aux_data[cc],
                )
                self._append_episodic_step(ep_step, step.agent_id[cc])

    def _append_episodic_step(self, ep_step: EpisodicMemoryStep, agent_id: Any):
        self._agent_step_counter += 1
        if agent_id not in self._active_episodes:
            self._active_episodes[agent_id] = StepMemory()
        self._active_episodes[agent_id].append(ep_step)

        if self._active_episodes[agent_id].count() > self._min_episode_length:
            if agent_id not in self._active_episodes_that_can_be_sampled.keys():
                self._active_episodes_that_can_be_sampled[agent_id] = self._active_episodes[agent_id]
                # print(f"adding id:{agent_id}: count: {self._active_episodes_that_can_be_sampled[agent_id].count()}")
            else:
                # print(f"id:{agent_id}: count: {self._active_episodes_that_can_be_sampled[agent_id].count()}, should be:{self._active_episodes[agent_id].count()}")
                pass

        if ep_step.done:
            completed_episode = self._active_episodes[agent_id]

            # Store the completed episode if it is greater than min size
            if completed_episode.count() > self._min_episode_length:
                self._completed_episodes.append(completed_episode)
            else:
                if self._debug:
                    print("episode too short")

            if agent_id in self._active_episodes_that_can_be_sampled.keys():
                self._active_episodes_that_can_be_sampled.pop(agent_id)

            # Create a new active episode
            self._active_episodes[agent_id] = StepMemory()

        if self.curr_memory_size() >= self._max_total_steps:
            # self._completed_episodes.popleft()
            self._completed_episodes.pop(0)

    def curr_memory_size(self) -> int:
        with self._lock:
            return sum([x.count() for x in self._completed_episodes])

    def num_completed_episodes(self) -> int:
        return len(self._completed_episodes)

    def num_completed_steps(self) -> int:
        with self._lock:
            return self._agent_step_counter

    def sample(self, batch_size: int, roll_length: int):
        with self._lock:
            num_rolls = batch_size // roll_length
            rolls = []
            num_active = len(self._active_episodes_that_can_be_sampled)
            num_complete = len(self._completed_episodes)
            total = num_active + num_complete
            fraction_active = num_active / total
            num_active_to_sample = int(num_rolls * fraction_active)
            num_complete_to_sample = int(num_rolls - num_active_to_sample)
            # print(num_rolls)
            # print(f" total:{total}, active: {num_active_to_sample}, complete: {num_complete_to_sample}")
            if (num_complete > 0) and (num_complete_to_sample > 0):
                episodes = self.sample_random_episodes(num_complete_to_sample)
                rolls.extend([ep.sample_rollout(roll_length) for ep in episodes])
            if (num_active > 0) and (num_active_to_sample > 0):
                episodes = self.sample_random_active_episodes(num_active_to_sample)
                rolls.extend([ep.sample_rollout(roll_length) for ep in episodes])
            return rolls

    def sample_random_episodes(self, num_episodes):
        indices = np.random.randint(0, self.num_completed_episodes(), num_episodes)
        episodes = [self._completed_episodes[cc] for cc in indices]
        return episodes

    def sample_random_active_episodes(self, num_episodes):
        indices = np.random.randint(0, len(self._active_episodes_that_can_be_sampled), num_episodes)
        actives = list(self._active_episodes_that_can_be_sampled.values())
        episodes = [actives[cc] for cc in indices]
        return episodes


class ThreadSafeBuffer:
    def __init__(self):
        self._lock = threading.RLock()
        # self._memory = deque()
        self._memory = []
        self._total_appends = 0

    def append(self, step: EpisodicMemoryStep):
        with self._lock:
            self._memory.append(step)
            self._total_appends += 1

    def count(self) -> int:
        with self._lock:
            return len(self._memory)

    def sample_all(self):
        with self._lock:
            samples = []
            while self.count() > 0:
                samples.append(self._memory.pop(0))
                # samples.append(self._memory.popleft())
            return samples

    def _random_location(self):
        with self._lock:
            return self._memory[np.random.randint(0, self.count())]

    def _random_locations(self, num_locations):
        with self._lock:
            samples = []
            if self.count() >= num_locations:
                for _ in range(num_locations):
                    samples.append(self._random_location())
            return samples

    def _random_slice(self, length):
        with self._lock:
            max_val = self.count() - length + 1
            start = np.random.randint(0, max_val)
            return self._memory[start : start + length]
            # return list(itertools.islice(self._memory, start, start+length))

    def __getitem__(self, key):
        with self._lock:
            return self._memory[key]


class StepMemory(ThreadSafeBuffer):
    def num_completed_steps(self) -> int:
        return self._total_appends

    def sample_steps_batch(self, batch_size: int):
        return self._random_locations(batch_size)

    def sample_rollout(self, roll_length: int) -> Roll:
        with self._lock:
            if self.count() >= roll_length:
                return Roll(steps=self._random_slice(roll_length), clone=True)
            else:
                assert False


class StepMemoryWithLimit(ThreadSafeBuffer):
    def __init__(self, max_size: int):
        super().__init__()
        self._max_size = max_size
        self._pointer = 0

    def append(self, step: EpisodicMemoryStep):
        with self._lock:
            if self.count() < self._max_size:
                self._memory.append(step)
            else:
                self._memory[self._pointer] = step
            self._pointer = (self._pointer + 1) % self._max_size
            self._total_appends += 1


class EpisodicBuffer(ThreadSafeBuffer):
    pass
