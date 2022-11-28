import threading
from time import sleep
from typing import Any
import gym

import torch
from agency.layers.feed_forward import InputNormalizer
from agency.memory.block_memory import AgentStep
from agency.worlds.gym_env import EpisodeInfo
from agency.worlds.gym_wrappers import BraxVectorGymWrapperWithHtmlRendering
import jax
from brax.envs import create as create_brax_env
from brax.envs import to_torch


class BraxGymVecEnvThread(threading.Thread):
    def __init__(
        self,
        thread_id: int,
        inferer: Any,
        memory: Any,
        params: Any,
        episode_info_buffer: Any,
        make_env_fn: Any,
        device: Any = "cuda",
    ):
        super().__init__()
        self._thread_id = thread_id
        self._inferer = inferer
        self._params = params
        self._memory = memory
        self._episode_info_buffer = episode_info_buffer
        self._num_steps = 10_000_000_000
        self._has_stopped = False
        self._should_stop = False
        self._should_pause = False
        self._device = device

    def request_stop(self, should_stop: bool = True):
        self._should_stop = should_stop

    def pause(self, should_pause: bool = True):
        self._should_pause = should_pause

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def _process_obs(self, obs: torch.Tensor):
        obs = obs.float()
        obs_norm = self._input_normalizer.normalize(obs)
        return obs_norm

    def run(self):
        num_workers = self._params.num_workers

        def make_env(render):
            environment = create_brax_env(env_name=self._params.name, batch_size=num_workers)
            gym_env = BraxVectorGymWrapperWithHtmlRendering(
                environment,
                seed=0,
                backend="gpu",
                render=render,
                episodes_per_render=self._params.episodes_per_render,
            )
            gym_env = to_torch.JaxToTorchWrapper(gym_env, device=self._device)
            return gym_env

        env = make_env(render=self._params.render)

        print("jit compiling env.reset")
        obs = env.reset()

        print("jit compiling env.step")
        action = torch.rand(env.action_space.shape, device=self._device) * 2 - 1
        obs, reward, done, info = env.step(action)

        self._input_normalizer = InputNormalizer(
            obs.shape[1:], clamp=True, clamp_val=self._params.obs_clamp, device=self._device
        )

        step_count = 0
        episode_count = torch.zeros(num_workers, device=self._device)
        episode_step_count = torch.zeros(num_workers, device=self._device)
        episode_reward = torch.zeros(num_workers, device=self._device)

        obs = self._process_obs(env.reset())
        while (step_count < self._num_steps) and not self._should_stop:
            while self._should_pause and not self._should_stop:
                sleep(1.0)
            step_count += num_workers
            episode_step_count += 1

            policy, aux_data = self._inferer.infer(obs)
            action = policy.sample

            (obs_next, reward, done, _) = env.step(action)

            obs_next = self._process_obs(obs_next)
            episode_reward += reward

            num_done = done.sum()
            if num_done > 0:
                mean_rewards = (episode_reward * done).sum() / num_done
                mean_steps = (episode_step_count * done).sum() / num_done
                self._episode_info_buffer.append(
                    EpisodeInfo(
                        reward=mean_rewards.cpu().numpy(),
                        steps=mean_steps.cpu().numpy(),
                    )
                )

            not_done = 1.0 - done
            episode_reward *= not_done
            episode_step_count *= not_done
            episode_count += done

            self._memory.append(
                step=AgentStep(
                    obs=obs,
                    obs_next=obs_next,
                    reward=reward,
                    policy=policy,
                    done=done,
                    aux_data=aux_data,
                    agent_id=list(range(num_workers)),
                )
            )
            obs = obs_next

        env.close()

        self._has_stopped = True
        print("worker finished.")
