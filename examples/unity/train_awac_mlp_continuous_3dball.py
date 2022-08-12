import random
from dataclasses import dataclass
from agency.algo.awac.network import AwacParams

import torch
from agency.algo import awac
from agency.algo.sac.network import MlpNetworkArchitecture
from agency.algo.sac.batch import create_batch
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.layers.distributions import ContinuousDistParams
from agency.core.logger import LogParams
from agency.memory import MemoryParams
from agency.worlds.simulator import create_unity_simulator
from agency.core.experiment import TrainLoopParams, start_experiment_helper
from agency.worlds.unity_env import UnityWorldParams


@dataclass
class WorldParams(UnityWorldParams):
    name: str = "3DBall"
    render: bool = False
    num_workers: int = 1
    input_size: int = 8
    num_actions: int = 2


class HyperParams:
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
        max_agent_steps=40_000,
    )

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[256, 256],
        p_hidden_sizes=[256, 256],
    )

    dist = ContinuousDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=10,
    )

    backprop = BackpropParams(
        batch_size=1000,
        clip_norm=10.0,
    )

    algo = AwacParams(learning_rate=0.003)

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"B_{self.algo.beta}_"
            + f"AC_{self.algo.adv_clip}_"
            + f"G_{self.rl.gamma}_"
            + f"SOFT_{self.algo.use_softmax}_"
            + f"CN_{self.backprop.clip_norm}"
        )

    def randomize(self, counter):
        self.algo.learning_rate = random.choice([0.001, 0.003, 0.006])
        self.algo.beta = random.choice([0.2])
        self.algo.adv_clip = random.choice([1, 2])
        self.algo.use_softmax = random.choice([False])


if __name__ == "__main__":
    start_experiment_helper(
        "3dBALL_AWAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=awac.network.create_continuous_network,
        create_train_data_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_unity_simulator,
    )
