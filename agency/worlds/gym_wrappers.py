from typing import Any, Callable, Optional
import os
import gym
import numpy as np
import pygame
from brax import jumpy as jp
from brax.envs import env as brax_env
from brax.envs import wrappers
from brax.io import html
from brax.io.file import File
from brax.io.file import MakeDirs
from gym import ObservationWrapper


class BraxVectorGymWrapperWithHtmlRendering(wrappers.VectorGymWrapper):
    def __init__(
        self,
        env: brax_env.Env,
        seed: int = 0,
        backend: Optional[str] = None,
        render: bool = True,
        episodes_per_render: int = 4,
    ):
        super().__init__(env, seed, backend)
        self._render = render
        self._episodes_per_render = episodes_per_render
        self._episode_counter = 0
        if self._render:
            self.episode_rollouts_agent0 = []

    def reset(self):
        self._state, obs, self._key = self._reset(self._key)
        if self._render:
            self.episode_rollouts_agent0 = [jp.take(self._state.qp, 0)]
        return obs

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)
        if self._render:
            self.episode_rollouts_agent0.append(jp.take(self._state.qp, 0))
            _truncated = info["truncation"][0] if "truncation" in info else False
            _done = done[0]
            if _done or _truncated:
                self._episode_counter += 1
                if self._episode_counter % self._episodes_per_render == 0:
                    self._save_episode_as_html()
                self.episode_rollouts_agent0 = []
        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def _save_episode_as_html(self):
        print("saving html episode")
        path = "/tmp/html/brax_episode.html"  # TODO: make this an input.
        sys = self._env.unwrapped.sys
        qps = self.episode_rollouts_agent0
        # html.save_html(path, sys, qps, make_dir=True)
        MakeDirs(os.path.dirname(path))
        with File(path, "w") as fout:
            fout.write(html.render(sys, qps, height=960))

    def close(self):
        pass


class PygameRgbArrayRenderer(gym.Wrapper):
    """Render rgb arrays using pygame.

    This is a simplified version of HumanRendering from openai gym, that works with atari.
    """

    def __init__(self, env):
        super().__init__(env)
        self.window_surface = None
        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.display.init()

    def render(self, **kwargs):
        rgb_array = self.env.render(mode="rgb_array")
        rgb_array = np.transpose(rgb_array, axes=(1, 0, 2))

        if self.window_surface is None:
            self.window_surface = pygame.display.set_mode(rgb_array.shape[:2])

        self.window_surface.blit(
            pygame.surfarray.make_surface(rgb_array),
            (0, 0),
        )
        pygame.event.pump()
        self.clock.tick()
        pygame.display.flip()

    def close(self):
        super().close()
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()


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
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, self.transpose_list)


class ChangeBoxObsSpaceDtypeAndTransformObs(ObservationWrapper):
    """Changes the data type of a Box observation and transforms the obs using input callable"""

    def __init__(self, env: gym.Env, dtype: Any, f: Callable[[Any], Any]):
        super().__init__(env)
        low = self.observation_space.low.astype(dtype)
        high = self.observation_space.high.astype(dtype)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=self.observation_space.shape, dtype=dtype
        )
        assert callable(f)
        self.f = f

    def observation(self, obs):
        return self.f(obs)
