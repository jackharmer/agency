import random
from dataclasses import dataclass

import torch
from agency.algo import awac
from agency.algo.awac.network import AwacParams
from agency.algo.sac.network import (ContinuousDistParams,
                                     MlpNetworkArchitecture)
from agency.core.batch import create_batch
from agency.core.experiment import start_experiment_helper
from agency.core.logger import LogParams
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams:
    name: str = "LunarLanderContinuous-v2"
    env_class: str = "box2d"
    render: bool = False
    is_image: bool = False
    num_workers: int = 10
    input_size: int = 8 * 3
    num_actions: int = 2
    use_vecenv: bool = True


class HyperParams:
    device = torch.device('cuda')

    log = LogParams(
        log_dir="/tmp/agency/logs/lunar_lander",
        train_samples_per_log=200_000
    )

    # memory
    max_agent_steps: int = 500_000
    max_memory_size: int = 1_000_000
    init_steps: int = 20_000

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
    batch_size: int = 2000
    learning_rate: float = 0.008
    clip_norm: float = 1.0

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
        self.learning_rate = random.choice([0.0001, 0.0005, 0.001, 0.003])
        self.roll_length = random.choice([5, 10, 20])
        self.batch_size = random.choice([500, 1000, 2000, 4000])
        self.algo.beta = random.choice([0.1, 0.2, 0.3])
        self.algo.adv_clip = random.choice([1, 2])
        self.algo.use_softmax = random.choice([False, True])
        self.gamma = random.choice([0.95, 0.97, 0.99])
        self.clip_norm = random.choice([10.0])


if __name__ == "__main__":
    start_experiment_helper(
        "LL_CONT_AWAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=awac.network.create_continuous_network,
        create_train_data_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )
