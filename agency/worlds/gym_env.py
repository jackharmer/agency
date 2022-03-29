import threading
from dataclasses import dataclass
from typing import Any

import gym
import numpy as np
import torch
from agency.memory.episodic import Step
from gym.vector.async_vector_env import AsyncVectorEnv
from gym_minigrid.wrappers import ImgObsWrapper
from gym import ObservationWrapper


class TransposeBoxObservation(ObservationWrapper):
    """Transpose a box observation"""

    def __init__(self, env, transpose_list):
        super().__init__(env)
        self.transpose_list = transpose_list

        new_shape = tuple(env.observation_space.shape[i] for i in transpose_list)
        new_low = env.observation_space.low.transpose(self.transpose_list)
        new_high = env.observation_space.high.transpose(self.transpose_list)
        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return np.transpose(obs, self.transpose_list)


@dataclass
class EpisodeInfo:
    reward: float
    steps: int


def gym_create_env_helper(env_class, name, is_image):
    env = gym.make(name)
    if env_class == "atari":
        env = gym.wrappers.AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=is_image
        )
        env = gym.wrappers.FrameStack(env, 3)

    elif env_class == "box2d":
        if is_image:
            env = gym.wrappers.ResizeObservation(env, shape=[84, 84])
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.array(obs).astype(np.float32) / 255.0
            )
        env = gym.wrappers.FrameStack(env, 3)
        if not is_image:
            env = gym.wrappers.FlattenObservation(env)

    elif env_class == "minigrid":
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        env = TransposeBoxObservation(env, [2, 0, 1])

    elif env_class == "gym-nowrappers":
        pass
    else:
        raise Exception("Unknown gym env class")

    print("Worker created gym env.")
    return env


class GymThread(threading.Thread):
    def __init__(
            self,
            thread_id: int,
            inferer: Any,
            memory: Any,
            params: Any,
            episode_info_buffer: Any,
            make_env_fn: Any
    ):
        super().__init__()
        self._thread_id = thread_id
        self._inferer = inferer
        self._params = params
        self._memory = memory
        self._episode_info_buffer = episode_info_buffer
        self._is_first_worker = self._thread_id == 0
        self._num_steps = 1_000_000_000
        self._has_stopped = False
        self._should_stop = False
        self._make_env_fn = make_env_fn

    def request_stop(self):
        self._should_stop = True

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def run(self):
        print(f"Worker {self._thread_id}: Making gym env.")
        env = self._make_env_fn(self._params.env_class, self._params.name, self._params.is_image)
        print(f"Worker {self._thread_id}: Created gym env.")

        step_count = 0
        episode_count = 0
        episode_step_count = 0
        episode_reward = 0.0
        reward = 0.0
        done = False

        obs = env.reset()

        while (step_count < self._num_steps) and not self._should_stop:
            step_count += 1
            episode_step_count += 1

            policy = self._inferer.infer(torch.tensor([obs], dtype=torch.float32), random_actions=False).numpy()
            action = policy.sample

            if type(env.action_space) == gym.spaces.Discrete:
                action = action.argmax(axis=1)
            (obs_next, reward, done, _) = env.step(action[0])

            if self._is_first_worker and self._params.render and (episode_count % self._params.episodes_per_render) == 0:
                env.render()

            step = Step(
                obs=np.array(obs, dtype=np.float32),
                policy=policy,
                reward=reward,
                done=done,
                obs_next=np.array(obs_next, dtype=np.float32)
            )
            obs = obs_next

            episode_reward = episode_reward + step.reward

            if step.done:
                obs = env.reset()
                self._episode_info_buffer.append(
                    EpisodeInfo(
                        reward=episode_reward,
                        steps=episode_step_count
                    )
                )
                episode_reward = 0.0
                episode_step_count = 0
                episode_count += 1
            self._memory.append(step=step, agent_id=self._thread_id)

        self._has_stopped = True
        print(f"worker {self._thread_id} finished.")


class GymVecEnvThread(threading.Thread):
    def __init__(
            self,
            thread_id: int,
            inferer: Any,
            memory: Any,
            params: Any,
            episode_info_buffer: Any,
            make_env_fn: Any
    ):
        super().__init__()
        self._thread_id = thread_id
        self._inferer = inferer
        self._params = params
        self._memory = memory
        self._episode_info_buffer = episode_info_buffer
        self._num_steps = 1_000_000_000
        self._has_stopped = False
        self._should_stop = False
        self._make_env_fn = make_env_fn

    def request_stop(self):
        self._should_stop = True

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def run(self):
        def make_env():
            return self._make_env_fn(self._params.env_class, self._params.name, self._params.is_image)

        num_workers = self._params.num_workers
        env_fns = [make_env for _ in range(num_workers)]

        is_discrete = False

        env = AsyncVectorEnv(env_fns, shared_memory=True)
        print("Action space: ", env.single_action_space)
        if isinstance(env.single_action_space, gym.spaces.Discrete):
            is_discrete = True

        step_count = 0
        episode_step_count = np.zeros(num_workers)
        episode_reward = np.zeros(num_workers)

        obs = env.reset()
        while (step_count < self._num_steps) and not self._should_stop:
            step_count += num_workers

            for cc in range(num_workers):
                episode_step_count[cc] += 1

            policy = self._inferer.infer(torch.tensor(obs, dtype=torch.float32), random_actions=False).numpy()
            action = policy.sample

            if is_discrete:
                action = action.argmax(axis=1)

            (obs_next, reward, done, _) = env.step(action)

            if self._params.render:
                env.render()
                env.ve

            policies = policy.split_batch_np()
            for cc in range(num_workers):
                step = Step(
                    obs=np.array(obs[cc], dtype=np.float32),
                    policy=policies[cc],
                    reward=reward[cc],
                    done=done[cc],
                    obs_next=np.array(obs_next[cc], dtype=np.float32)
                )

                episode_reward[cc] += step.reward

                if step.done:
                    self._episode_info_buffer.append(
                        EpisodeInfo(
                            reward=episode_reward[cc],
                            steps=episode_step_count[cc]
                        )
                    )
                    episode_reward[cc] = 0.0
                    episode_step_count[cc] = 0
                self._memory.append(step=step, agent_id=cc)

            obs = obs_next

        env.close()

        self._has_stopped = True
        print("worker finished.")
