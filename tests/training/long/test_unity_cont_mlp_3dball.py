from dataclasses import dataclass
import numpy as np

import torch
from agency.algo import sac, ppo
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
from agency.memory import MemoryParams, create_episodic_memory
from agency.worlds.simulator import create_unity_simulator
from agency.core.experiment import TrainLoopParams, start_experiment
from agency.worlds.unity_env import UnityWorldParams


@dataclass
class WorldParams(UnityWorldParams):
    name: str = "3DBall"
    num_workers: int = 1
    input_size: int = 8
    num_actions: int = 2


class SacHyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/3dball",
        train_samples_per_log=50_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000_000,
    )

    data = DataCollectionParams(
        init_agent_steps=5_000,
        init_episodes=1,
        max_agent_steps=20_000,
    )

    arch = sac.network.MlpNetworkArchitecture(
        q_hidden_sizes=[256, 256],
        p_hidden_sizes=[256, 256],
    )

    dist = sac.network.ContinuousDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=5,
    )

    backprop = BackpropParams(
        batch_size=1000,
        clip_norm=1.0,
    )

    algo = sac.network.SacParams(
        learning_rate=0.001,
        init_alpha=0.1,
    )


class PpoHyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/3dball",
        train_samples_per_log=20_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_500,
    )

    data = DataCollectionParams(
        init_agent_steps=1_000,
        init_episodes=1,
        max_agent_steps=20_000,
    )
    arch = ppo.network.MlpNetworkArchitecture(
        shared_hidden_sizes=None,
        v_hidden_sizes=[256, 256],
        p_hidden_sizes=[256, 256],
    )

    dist = ppo.network.ContinuousDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=10,
    )

    backprop = BackpropParams(
        batch_size=1000,
        clip_norm=5.0,
    )

    algo = ppo.network.PpoParams(
        p_learning_rate=0.0002,
        v_learning_rate=0.0002,
        normalize_advantage=False,
        clip_value_function=False,
        entropy_loss_scaling=0.0,
        v_clip=0.2,
        ppo_clip=0.2,
    )


def setup():
    torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(42)
    np.random.seed(42)


def test_sac_cont_3dball():
    setup()

    ep_info_buffer = start_experiment(
        hp=SacHyperParams(),
        wp=WorldParams(),
        exp_name="SacContUnity3dballTest",
        network_creater_fn=sac.network.create_continuous_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_unity_simulator,
        create_memory_fn=create_episodic_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 10.0


def test_ppo_cont_3dball():
    setup()

    ep_info_buffer = start_experiment(
        hp=PpoHyperParams(),
        wp=WorldParams(),
        exp_name="PpoContUnity3dballTest",
        network_creater_fn=ppo.network.create_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_unity_simulator,
        create_memory_fn=create_episodic_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 10.0
