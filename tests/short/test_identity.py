from dataclasses import dataclass

import agency.algo.awac as awac
import agency.algo.sac as sac
from agency.core.logger import LogParams
import agency.worlds.games.gym  # This import is required, it registers the environments.
import gym
import numpy as np
import torch
from agency.algo.awac.network import AwacParams
from agency.algo.sac.network import (ContinuousDistParams, DiscreteDistParams,
                                     MlpNetworkArchitecture, SacParams)
from agency.core.batch import create_batch, create_batch_categorical
from agency.core.experiment import start_experiment
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams:
    name: str = 'discrete-identity-env-v0'
    env_class: str = "gym-nowrappers"
    render: bool = False
    episodes_per_render: int = 1
    input_size: int = 3
    is_image: bool = False
    num_workers: int = 1
    num_actions: int = 3
    use_vecenv: bool = True


class BaseHyperParams:
    device = torch.device('cpu')

    log = LogParams(
        log_dir="/tmp/agency/tests/identity",
        train_samples_per_log=1_000
    )

    max_agent_steps: int = 4_000

    # memory
    max_memory_size: int = 1_000_000
    init_steps: int = 100

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[100, 100],
        p_hidden_sizes=[100, 100]
    )

    # rl
    gamma: float = 0.99
    roll_length: int = 1
    reward_scaling: float = 1.0
    reward_clip_value: float = 1000.0

    # training
    batch_size: int = 100
    learning_rate: float = 0.003
    clip_norm: float = 10.0


class SacCategoricalParams(BaseHyperParams):
    algo = SacParams(
        target_entropy_constant=0.2
    )
    dist = DiscreteDistParams(
        categorical=True
    )


class SacGumbelParams(BaseHyperParams):
    algo = SacParams()
    dist = DiscreteDistParams(
        categorical=False
    )


class SacContinuousParams(BaseHyperParams):
    algo = SacParams()
    dist = ContinuousDistParams()


class AwacContinuousParams(BaseHyperParams):
    algo = AwacParams()
    dist = ContinuousDistParams()


class AwacGumbelParams(BaseHyperParams):
    algo = AwacParams()
    dist = DiscreteDistParams(
        categorical=False
    )


def setup():
    torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(42)
    np.random.seed(42)


def test_categorical_sac():
    setup()

    ep_info_buffer = start_experiment(
        hp=SacCategoricalParams(),
        wp=WorldParams(),
        exp_name="CategoricalIdentityTestSAC",
        network_creater_fn=sac.network.create_discrete_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch_categorical,
        train_on_batch_fn=sac.trainer.train_on_batch_categorical,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_gumbel_sac():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=SacGumbelParams(),
        exp_name="GumbelIdentityTestSAC",
        network_creater_fn=sac.network.create_discrete_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_sac():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=SacContinuousParams(),
        exp_name="ContinuousIdentityTestSAC",
        network_creater_fn=sac.network.create_continuous_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_awac():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=AwacContinuousParams(),
        exp_name="ContinuousIdentityTestAWAC",
        network_creater_fn=awac.network.create_continuous_network,
        train_data_creator_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_gumbel_awac():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=AwacGumbelParams(),
        exp_name="GumbelIdentityTestAWAC",
        network_creater_fn=awac.network.create_discrete_network,
        train_data_creator_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7

