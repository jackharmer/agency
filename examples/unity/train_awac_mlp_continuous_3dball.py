import random
from dataclasses import dataclass
from agency.algo.awac.network import AwacParams

import torch
from agency.algo import awac
from agency.algo.sac.network import (ContinuousDistParams,
                                     MlpNetworkArchitecture)
from agency.core.batch import create_batch
from agency.core.logger import LogParams
from agency.worlds.simulator import create_unity_simulator
from agency.core.experiment import start_experiment_helper


@dataclass
class WorldParams:
    name: str = "3DBall"
    render: bool = False
    is_image: bool = False
    random_actions: bool = False
    num_workers: int = 1

    input_size: int = 8
    num_actions: int = 2
    use_registry: bool = True


class HyperParams:
    device = torch.device('cuda')

    log = LogParams(
        log_dir="/tmp/agency/logs/3dball",
        train_samples_per_log=50_000
    )

    # memory
    max_agent_steps: int = 40_000
    max_memory_size: int = 1_000_000
    init_steps: int = 5000

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[256, 256],
        p_hidden_sizes=[256, 256]
    )

    dist = ContinuousDistParams()

    # rl
    gamma: float = 0.99
    roll_length: int = 10
    reward_scaling: float = 1.0
    reward_clip_value: float = 10000.0

    # training
    batch_size: int = 1000
    learning_rate: float = 0.003
    clip_norm: float = 10.0

    algo = AwacParams()

    def get_fname_string_from_params(self):
        return f"LR_{self.learning_rate}_" + \
               f"R_{self.roll_length}_" + \
               f"BS_{self.batch_size}_" + \
               f"B_{self.algo.beta}_" + \
               f"AC_{self.algo.adv_clip}_" + \
               f"G_{self.gamma}_" + \
               f"SOFT_{self.algo.use_softmax}_" + \
               f"CN_{self.clip_norm}"

    def randomize(self, counter):
        self.learning_rate = random.choice([0.001, 0.003, 0.006])
        self.roll_length = random.choice([5, 10])
        self.batch_size = random.choice([1000, 2000, 4000])
        self.algo.beta = random.choice([0.2])
        self.algo.adv_clip = random.choice([1, 2])
        self.algo.use_softmax = random.choice([False])
        self.gamma = random.choice([0.99])
        self.clip_norm = random.choice([10.0])


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
        create_simulator_fn=create_unity_simulator
    )
