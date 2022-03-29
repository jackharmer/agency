import numpy as np
import gym
from gym import spaces


class IdentityEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_actions=3, game_length=10, is_continuous=False):
        super().__init__()
        self._game_length = game_length
        self._num_actions = num_actions
        self._is_continuous = is_continuous

        if self._is_continuous:
            self.action_space = spaces.Box(low=0, high=1.0, shape=(num_actions,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(num_actions)

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(num_actions,), dtype=np.float32)

        self._debug = False
        self._counter = 0
        self._active_index = 0
        self._obs = None

    def step(self, action):
        if self._is_continuous:
            reward = 0.0
            for cc in range(self._num_actions):
                if cc == self._active_index:
                    reward += np.abs(action[cc])
                else:
                    reward -= np.abs(action[cc])
        else:
            if action == self._active_index:
                reward = 1.0
            else:
                reward = 0.0

        self._counter += 1
        done = self._counter == self._game_length

        if self._debug:
            print(f"obs: {self._obs}, action: {action}, reward: {reward}, done: {done}, counter: {self._counter-1}, active_index: {self._active_index}")

        self._active_index = self._counter % self._num_actions
        self._obs = self._create_observation(done)
        return self._obs, reward, done, {}

    def reset(self):
        self._counter = 0
        self._active_index = 0
        self._obs = self._create_observation(False)
        return self._obs

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _create_observation(self, done):
        if done:
            obs = np.array([-1.0] * self._num_actions)
        else:
            obs = np.array([0.0] * self._num_actions)
            obs[self._active_index] = 1.0
        return obs


class IdentityEnvContinuous(IdentityEnv):
    def __init__(self, num_actions=3, game_length=10):
        super().__init__(num_actions=num_actions, game_length=game_length, is_continuous=True)
