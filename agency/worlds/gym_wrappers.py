from typing import Any, Callable
import gym
import numpy as np
import pygame
from gym import ObservationWrapper


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
