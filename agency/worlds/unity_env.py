import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from agency.memory.block_memory import AgentStep
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple


@dataclass
class EpisodeInfo:
    reward: float
    steps: int


@dataclass
class UnityWorldParams:
    name: str
    input_size: int
    num_actions: int
    render: bool = False
    is_image: bool = False
    num_workers: int = 1
    random_actions: bool = False
    use_registry: bool = True


class AgentData:
    def __init__(self):
        self.last_obs = None
        self.last_policy = None
        self.last_aux_data = None
        self.total_reward = 0
        self.step_count = 0

    def has_previous_obs(self):
        return self.last_obs is not None


class UnityThread(threading.Thread):
    """
    Basic mlagents data collection. This is missing a lot of functionality. For instance action branching.
    """

    def __init__(
        self,
        thread_id: int,
        inferer: Any,
        memory: Any,
        params: Any,
        episode_info_buffer: Any,
        make_env_fn: Any = None,  # API requirement, currently unused here.
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

    def request_stop(self):
        self._should_stop = True

    def stop(self):
        pass

    def is_ready_to_stop(self):
        return self._has_stopped

    def get_memory_id(self, local_agent_id):
        return [(self._thread_id << 12) + local_agent_id]

    def run(self):
        print(f"Worker {self._thread_id}: Making unity env.")
        # environment_names = list(default_registry.keys())
        # for name in environment_names:
        #     print(name)

        channel = EngineConfigurationChannel()
        worker_id = 10 + self._thread_id

        if self._params.use_registry:
            env = default_registry[self._params.name].make(worker_id=worker_id, side_channels=[channel])
        else:
            if self._params.name is None:
                worker_id = 0
            env = UnityEnvironment(
                file_name=self._params.name,  # Use file_name=None for PIE
                worker_id=worker_id,
                side_channels=[channel],
                no_graphics=False,
                additional_args=["-logFile", "-"],
            )

        print(f"Worker {self._thread_id}: Created unity env.")
        env.reset()

        # channel.set_configuration_parameters(time_scale=1.0)

        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        discrete_action_space = spec.action_spec.is_discrete()
        # continuous_action_space = spec.action_spec.is_continuous()
        for obs_spec in spec.observation_specs:
            print(f"Observation shape: {obs_spec.shape}")
            print(f"Observation type: {obs_spec.observation_type}")
            # print(f"Observation count: {obs_spec.count()}")

        print(f"Behavior name: {behavior_name}")
        print(f"Is continuous actions : {spec.action_spec.is_continuous()}")
        print(f"Action branches: {spec.action_spec.discrete_branches}")
        print(f"Discrete size: {spec.action_spec.discrete_size}")
        print(f"Continuous size: {spec.action_spec.continuous_size}")

        step_count = 0
        agent_data_dict: dict[int, AgentData] = {}

        def get_obs0(agent, permute=True):
            obs = agent.obs[0]
            if self._params.is_image and permute:
                obs = np.swapaxes(obs, 0, 2)
            obs = torch.from_numpy(obs).float()
            return obs

        while (step_count < self._num_steps) and not self._should_stop:
            active_agents, terminal_agents = env.get_steps(behavior_name)
            step_count += len(active_agents)

            # TERMINAL AGENTS
            for agent_id in terminal_agents:
                agent = terminal_agents[agent_id]
                agent_data = agent_data_dict[agent_id]

                obs = get_obs0(agent)

                final_step = AgentStep(
                    obs=agent_data.last_obs.unsqueeze(0),  # .copy(),
                    obs_next=obs.unsqueeze(0),
                    reward=torch.tensor(agent.reward).unsqueeze(0),
                    policy=[agent_data.last_policy],  # .copy(),
                    done=torch.tensor(not agent.interrupted).unsqueeze(0),
                    aux_data=[agent_data.last_aux_data],
                    agent_id=self.get_memory_id(agent_id),
                )
                self._memory.append(step=final_step)

                final_reward = agent_data.total_reward + agent.reward
                step_count = agent_data.step_count
                self._episode_info_buffer.append(
                    EpisodeInfo(
                        reward=final_reward,
                        steps=step_count,
                    )
                )
                agent_data_dict.pop(agent_id)

            # ACTIVE AGENTS
            for agent_id in active_agents:
                agent = active_agents[agent_id]
                if agent_id not in agent_data_dict:
                    agent_data_dict[agent_id] = AgentData()
                agent_data = agent_data_dict[agent_id]

                obs = get_obs0(agent)
                if agent_data.has_previous_obs():
                    step = AgentStep(
                        obs=agent_data.last_obs.unsqueeze(0),  # .copy(),
                        obs_next=obs.unsqueeze(0),
                        reward=torch.tensor(agent.reward).unsqueeze(0),
                        policy=[agent_data.last_policy],  # .copy(),
                        done=torch.tensor(False).unsqueeze(0),
                        aux_data=[agent_data.last_aux_data],
                        agent_id=self.get_memory_id(agent_id),
                    )
                    # add a step to memory
                    self._memory.append(step=step)

                    agent_data.total_reward += agent.reward

                agent_data.last_obs = obs
                agent_data.step_count += 1

            # INFERENCE
            if active_agents:
                obs = get_obs0(active_agents, permute=False)
                if self._params.is_image:
                    obs = obs.permute(0, 3, 1, 2)

                policy, aux_data = self._inferer.infer(obs, self._params.random_actions)

                action_tuple = ActionTuple()
                if discrete_action_space:
                    action = policy.sample.cpu().numpy().argmax(axis=1).reshape(-1, 1)
                    action_tuple.add_discrete(action)
                else:
                    action_tuple.add_continuous(policy.sample.cpu().numpy())
                env.set_actions(behavior_name, action_tuple)

                for agent_index, agent_id in enumerate(active_agents.agent_id):
                    agent_data_dict[agent_id].last_policy = policy[agent_index]
                    agent_data_dict[agent_id].last_aux_data = aux_data[agent_index]

            # STEP WORLD
            env.step()

        env.close()
        time.sleep(5)

        self._has_stopped = True
        print(f"worker {self._thread_id} finished.")
