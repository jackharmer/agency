from dataclasses import dataclass
import numpy as np

import torch
from agency.algo import sac
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
from agency.memory import MemoryParams, create_episodic_memory
from agency.worlds.simulator import create_unity_simulator
from agency.core.experiment import TrainLoopParams, start_experiment
from agency.worlds.unity_env import UnityWorldParams


@dataclass
class WorldParams(UnityWorldParams):
    name: str = "Basic"
    num_workers: int = 1
    input_size: int = 20
    num_actions: int = 3


class SacHyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/basic",
        train_samples_per_log=50_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000_000,
    )

    data = DataCollectionParams(
        init_agent_steps=1_000,
        init_episodes=1,
        max_agent_steps=5_000,
    )

    arch = sac.network.MlpNetworkArchitecture(
        q_hidden_sizes=[64, 64],
        p_hidden_sizes=[64, 64],
    )

    dist = sac.network.DiscreteDistParams(temperature=2.0)

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=2,
    )

    backprop = BackpropParams(
        batch_size=1000,
        clip_norm=1.0,
    )

    algo = sac.network.SacParams(learning_rate=0.001)


def setup():
    torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(42)
    np.random.seed(42)


def test_sac_gumbel():
    setup()

    ep_info_buffer = start_experiment(
        hp=SacHyperParams(),
        wp=WorldParams(),
        exp_name="SacGumbelUnityBasicTest",
        network_creater_fn=sac.network.create_discrete_network,
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
    assert mean_reward > 0.70
