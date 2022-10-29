import threading
from dataclasses import dataclass
from functools import partial
from time import sleep
from typing import Any

import gym
import numpy as np
import torch
from agency.memory.block_memory import AgentStep
from agency.memory.episodic import EpisodicMemoryStep
from agency.worlds.gym_wrappers import (
    ChangeBoxObsSpaceDtypeAndTransformObs,
    PygameRgbArrayRenderer,
    TransposeBoxObservation,
)
from gym.error import AlreadyPendingCallError
from gym.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gym_minigrid.wrappers import ImgObsWrapper


class RenderableAsyncVectorEnv(AsyncVectorEnv):
    """Allows rendering of AsyncVectorEnv envs"""

    def render_one_env(self, env_id: int, *args, **kwargs) -> list[Any]:
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(f"Waiting for a pending call to to complete.", self._state.value)
        self.parent_pipes[env_id].send(("_call", ("render", args, kwargs)))
        self._state = AsyncState.WAITING_CALL
        self.parent_pipes[env_id].recv()
        self._state = AsyncState.DEFAULT


@dataclass
class EpisodeInfo:
    reward: float
    steps: int
    video_data: np.ndarray = None


@dataclass
class GymWorldParams:
    env_class: str
    name: str
    input_size: int
    num_actions: int
    is_image: bool = False
    render: bool = False
    episodes_per_render: int = 1
    obs_scaling: float = None
    num_workers: int = 1
    use_vecenv: bool = True
    use_envpool: bool = False
    frame_stack: int = 3
    screen_size: int = 84


def gym_create_single_env(wp: GymWorldParams, render: bool = False):
    if wp.env_class == "atari":
        env = gym.make(
            wp.name,
            repeat_action_probability=0.0,
            obs_type="rgb",
            disable_env_checker=True,
        )
        if render:
            env = PygameRgbArrayRenderer(env)
        env = gym.wrappers.AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=4,
            screen_size=wp.screen_size,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False,
        )
        env = gym.wrappers.FrameStack(env, wp.frame_stack)
        env = ChangeBoxObsSpaceDtypeAndTransformObs(env, np.float32, lambda o: np.array(o, dtype=np.float32))
        if wp.obs_scaling is not None:
            env = gym.wrappers.TransformObservation(env, lambda obs: obs * wp.obs_scaling)

    elif wp.env_class == "box2d":
        env = gym.make(wp.name)
        if wp.is_image:
            env = gym.wrappers.ResizeObservation(env, shape=[wp.screen_size, wp.screen_size])
            env = gym.wrappers.GrayScaleObservation(env)

        if wp.obs_scaling is not None:
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.array(obs, dtype=np.float32) * wp.obs_scaling
            )

        env = gym.wrappers.FrameStack(env, wp.frame_stack)
        if not wp.is_image:
            env = gym.wrappers.FlattenObservation(env)

    elif wp.env_class == "minigrid":
        env = gym.make(wp.name, disable_env_checker=True)
        if render:
            env = PygameRgbArrayRenderer(env)  # Much more performant than their own rendering solution
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        env = TransposeBoxObservation(env, [2, 0, 1])

        if wp.obs_scaling is not None:
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.array(obs, dtype=np.float32) * wp.obs_scaling
            )

    elif wp.env_class == "gym-nowrappers":
        env = gym.make(wp.name)
    else:
        raise Exception("Unknown gym env class")

    # print("Worker created gym env.")
    return env


def gym_create_envpool_vecenv(wp: GymWorldParams):
    import envpool

    if wp.env_class == "atari":
        # see https://envpool.readthedocs.io/en/latest/env/atari.html#env-wrappers
        env = envpool.make(
            wp.name,
            env_type="gym",
            num_envs=wp.num_workers,
            episodic_life=True,
            reward_clip=False,
            stack_num=wp.frame_stack,
            gray_scale=True,
            frame_skip=4,
            noop_max=30,
            zero_discount_on_life_loss=False,
            img_height=wp.screen_size,
            img_width=wp.screen_size,
            repeat_action_probability=0.0,
            use_inter_area_resize=True,
        )
        if wp.obs_scaling is not None:
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.array(obs, dtype=np.float32) * wp.obs_scaling
            )
    else:
        raise Exception("Unsupported env class")

    env.num_envs = wp.num_workers
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)
    return env, is_discrete


def gym_create_async_vecenv(wp: GymWorldParams):
    def make_env(render):
        return gym_create_single_env(wp, render=render)

    env_fns = [partial(make_env, render=(cc == 0)) for cc in range(wp.num_workers)]
    env = RenderableAsyncVectorEnv(env_fns, shared_memory=True)
    is_discrete = isinstance(env.single_action_space, gym.spaces.Discrete)
    return env, is_discrete


class GymThread(threading.Thread):
    def __init__(
        self,
        thread_id: int,
        inferer: Any,
        memory: Any,
        params: Any,
        episode_info_buffer: Any,
        make_env_fn: Any,
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
        self._should_pause = False
        self._make_env_fn = make_env_fn

    def request_stop(self, should_stop: bool = True):
        self._should_stop = should_stop

    def pause(self, should_pause: bool):
        self._should_pause = should_pause

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def run(self):
        # print(f"Worker {self._thread_id}: Making gym env.")
        env = self._make_env_fn(
            self._params,
            render=self._is_first_worker and self._params.render,
        )
        # print(f"Worker {self._thread_id}: Created gym env.")

        step_count = 0
        episode_count = 0
        episode_step_count = 0
        episode_reward = 0.0
        reward = 0.0
        done = False

        obs = env.reset()
        obs = torch.from_numpy(np.array(obs)).float().unsqueeze(0)

        while (step_count < self._num_steps) and not self._should_stop:
            while self._should_pause and not self._should_stop:
                sleep(1.0)
            step_count += 1
            episode_step_count += 1

            policy, aux_data = self._inferer.infer(obs)
            action = policy.sample.cpu().numpy()

            if type(env.action_space) == gym.spaces.Discrete:
                action = action.argmax(axis=1)
            (obs_next, reward, done, _) = env.step(action[0])
            obs_next = torch.from_numpy(np.array(obs_next)).float().unsqueeze(0)

            if (
                self._is_first_worker
                and self._params.render
                and (episode_count % self._params.episodes_per_render) == 0
            ):
                env.render(mode="human")

            step = EpisodicMemoryStep(
                obs=obs[0],
                policy=policy,
                reward=reward,
                done=done,
                obs_next=obs_next[0],
                aux_data=aux_data,
            )
            obs = obs_next

            episode_reward = episode_reward + step.reward

            if step.done:
                obs = env.reset()
                obs = torch.from_numpy(np.array(obs)).float().unsqueeze(0)
                self._episode_info_buffer.append(
                    EpisodeInfo(
                        reward=episode_reward,
                        steps=episode_step_count,
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
        make_env_fn: Any,
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
        self._should_pause = False
        self._make_env_fn = make_env_fn

    def request_stop(self, should_stop: bool = True):
        self._should_stop = should_stop

    def pause(self, should_pause: bool = True):
        self._should_pause = should_pause

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def run(self):
        env, is_discrete = self._make_env_fn(self._params)

        num_workers = self._params.num_workers
        step_count = 0
        episode_count = torch.zeros(num_workers)
        episode_step_count = torch.zeros(num_workers)
        episode_reward = torch.zeros(num_workers)

        obs = env.reset()
        obs = torch.from_numpy(obs).float()
        while (step_count < self._num_steps) and not self._should_stop:
            while self._should_pause and not self._should_stop:
                sleep(1.0)
            step_count += num_workers
            episode_step_count += 1

            policy, aux_data = self._inferer.infer(obs)
            action = policy.sample.cpu().numpy()

            if is_discrete:
                action = action.argmax(axis=1)

            (obs_next, reward, done, _) = env.step(action)
            obs_next = torch.from_numpy(obs_next).float()
            reward = torch.from_numpy(reward).float()
            done = torch.from_numpy(done)

            episode_reward += reward

            if self._params.render and (episode_count[0] % self._params.episodes_per_render) == 0:
                env.render_one_env(env_id=0, mode="human")

            for cc in range(num_workers):
                if done[cc]:
                    self._episode_info_buffer.append(
                        EpisodeInfo(
                            reward=episode_reward[cc].clone().numpy(),
                            steps=episode_step_count[cc].clone().numpy(),
                        )
                    )
                    episode_reward[cc] = 0.0
                    episode_step_count[cc] = 0
                    episode_count[cc] += 1

            self._memory.append(
                step=AgentStep(
                    obs=obs,
                    obs_next=obs_next,
                    reward=reward,
                    policy=policy,
                    done=done.float(),
                    aux_data=aux_data,
                    agent_id=list(range(num_workers)),
                )
            )

            obs = obs_next

        env.close()

        self._has_stopped = True
        print("worker finished.")
