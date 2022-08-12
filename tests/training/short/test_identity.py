from dataclasses import dataclass

import agency.algo.awac as awac
import agency.algo.sac as sac
import agency.algo.ppo as ppo
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
from agency.memory import MemoryParams, create_episodic_memory, episodic
import agency.worlds.games.gym  # This import is required, it registers the environments.
import gym
import numpy as np
import torch
from agency.core.experiment import TrainLoopParams, start_experiment
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams(GymWorldParams):
    name: str = "discrete-identity-env-v0"
    env_class: str = "gym-nowrappers"
    render: bool = False
    input_size: int = 3
    num_workers: int = 1
    num_actions: int = 3


class BaseSacHyperParams:
    device = torch.device("cpu")

    log = LogParams(
        log_dir="/tmp/agency/tests/identity",
        train_samples_per_log=1_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000_000,
    )

    data = DataCollectionParams(
        init_agent_steps=100,
        init_episodes=1,
        max_agent_steps=4_000,
    )

    arch = sac.network.MlpNetworkArchitecture(
        q_hidden_sizes=[100, 100],
        p_hidden_sizes=[100, 100],
    )

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=2,
    )

    backprop = BackpropParams(
        batch_size=100,
        clip_norm=10.0,
    )


class BasePpoParams:
    device = torch.device("cpu")

    log = LogParams(
        log_dir="/tmp/agency/tests/identity",
        train_samples_per_log=2_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000,
    )

    data = DataCollectionParams(
        init_agent_steps=50,
        init_episodes=1,
        max_agent_steps=8_000,
    )

    arch = ppo.network.MlpNetworkArchitecture(
        shared_hidden_sizes=None,
        v_hidden_sizes=[100, 100],
        p_hidden_sizes=[100, 100],
    )

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=2,
    )

    backprop = BackpropParams(
        batch_size=100,
        clip_norm=5.0,
    )

    algo = ppo.network.PpoParams(
        p_learning_rate=0.0005,
        v_learning_rate=0.0005,
        normalize_advantage=False,
        clip_value_function=False,
        entropy_loss_scaling=0.0,
    )


class PpoContinuousParams(BasePpoParams):
    dist = ppo.network.ContinuousDistParams()


class PpoGumbelParams(BasePpoParams):
    dist = ppo.network.DiscreteDistParams()


class PpoCategoricalParams(BasePpoParams):
    dist = ppo.network.DiscreteDistParams(categorical=True)


class SacCategoricalParams(BaseSacHyperParams):
    algo = sac.network.SacParams(
        learning_rate=0.003,
        target_entropy_constant=0.2,
    )
    dist = sac.network.DiscreteDistParams(categorical=True)


class SacGumbelParams(BaseSacHyperParams):
    algo = sac.network.SacParams(learning_rate=0.003)
    dist = sac.network.DiscreteDistParams(categorical=False)


class SacContinuousParams(BaseSacHyperParams):
    algo = sac.network.SacParams(learning_rate=0.003)
    dist = sac.network.ContinuousDistParams()


class AwacContinuousParams(BaseSacHyperParams):
    algo = awac.network.AwacParams(learning_rate=0.003)
    dist = sac.network.ContinuousDistParams()


class AwacGumbelParams(BaseSacHyperParams):
    algo = awac.network.AwacParams(learning_rate=0.003)
    dist = sac.network.DiscreteDistParams(categorical=False)


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
        create_batch_fn=sac.batch.create_categorical_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch_categorical,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
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
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_sac_episodic_memory():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=SacContinuousParams(),
        exp_name="ContinuousIdentityTestUsingEpisodicMemorySAC",
        network_creater_fn=sac.network.create_continuous_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=create_episodic_memory,
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
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
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
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
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
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_ppo():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=PpoContinuousParams(),
        exp_name="ContinuousIdentityTestPpo",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=ppo.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_ppo_episodic_memory():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=PpoContinuousParams(),
        exp_name="ContinuousIdentityTestUsingEpisodicMemoryPpo",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=create_episodic_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_continuous_dual_optim_ppo():
    setup()

    hp = PpoContinuousParams()
    hp.algo.use_dual_optimizer = True

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=hp,
        exp_name="ContinuousDualOptimIdentityTestPpo",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=ppo.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_gumbel_ppo():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=PpoGumbelParams(),
        exp_name="GumbelIdentityTestPpo",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=ppo.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7


def test_categorical_ppo():
    setup()

    ep_info_buffer = start_experiment(
        wp=WorldParams(),
        hp=PpoCategoricalParams(),
        exp_name="CategoricalIdentityTestPpo",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=ppo.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 7
