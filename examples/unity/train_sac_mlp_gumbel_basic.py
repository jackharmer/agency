import random
from dataclasses import dataclass

import torch
from agency.algo import sac
from agency.algo.sac.network import (DiscreteDistParams,
                                     MlpNetworkArchitecture, SacParams)
from agency.core.batch import create_batch
from agency.core.logger import LogParams
from agency.worlds.simulator import create_unity_simulator
from agency.core.experiment import start_experiment_helper


@dataclass
class WorldParams:
    name: str = "Basic"
    is_image: bool = False
    random_actions: bool = False
    num_workers: int = 1
    input_size: int = 20
    num_actions: int = 3
    use_registry: bool = True


class HyperParams:
    device = torch.device('cuda')

    log = LogParams(
        log_dir="/tmp/agency/logs/basic",
        train_samples_per_log=50_000
    )

    # memory
    max_agent_steps: int = 40_000
    max_memory_size: int = 1_000_000
    init_steps: int = 1000

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[64, 64],
        p_hidden_sizes=[64, 64]
    )

    dist = DiscreteDistParams(
        temperature=2.0
    )

    # RL
    gamma: float = 0.99
    roll_length: int = 2
    reward_scaling: float = 1.0
    reward_clip_value: float = 10000.0

    # TRAINING
    batch_size: int = 1000
    learning_rate: float = 0.001
    clip_norm: float = 1.0

    algo = SacParams()

    def get_fname_string_from_params(self):
        return f"LR_{self.learning_rate}_" + \
               f"R_{self.roll_length}_" + \
               f"BS_{self.batch_size}_" + \
               f"IA_{self.algo.init_alpha}_" + \
               f"TE_{self.algo.target_entropy_constant}_" + \
               f"G_{self.gamma}_" + \
               f"CN_{self.clip_norm}_" + \
               f"HS_{self.arch.q_hidden_sizes}"

    def randomize(self, counter):
        self.learning_rate = random.choice([0.0001, 0.0005, 0.001, 0.003])
        self.roll_length = random.choice([5, 10, 20])
        self.batch_size = random.choice([500, 1000, 2000, 4000])
        self.algo.init_alpha = random.choice([0.1, 0.2, 0.3])
        self.algo.target_entropy_constant = random.choice([0.7, 0.8, 1.0])
        self.gamma = random.choice([0.95, 0.97, 0.99])
        self.clip_norm = random.choice([1.0, 10.0, 100.0])


if __name__ == "__main__":
    start_experiment_helper(
        "UNITY_BASIC_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_discrete_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_unity_simulator
    )
