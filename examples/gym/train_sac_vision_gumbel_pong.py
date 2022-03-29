from dataclasses import dataclass
from typing import Tuple
from agency.core.logger import LogParams

import torch
from agency.algo import sac
from agency.algo.sac.network import (DiscreteDistParams, SacParams,
                                     VisionNetworkArchitecture)
from agency.core.batch import create_batch
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import start_experiment_helper

import gym


@dataclass
class WorldParams:
    name: str = "PongDeterministic-v4"
    env_class: str = "atari"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (84, 84, 3)
    render: bool = False
    episodes_per_render: int = 3
    num_workers: int = 6
    num_actions: int = 6
    use_vecenv: bool = True


class HyperParams:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log = LogParams(
        log_dir="/tmp/agency/logs/pong",
        train_samples_per_log=100_000
    )

    max_agent_steps: int = 1_000_000
    max_memory_size: int = 60_000
    init_steps: int = 2_000

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[16, 16, 32, 32],
        shared_enc_kernel_sizes=[3, 3, 3, 3],
        shared_enc_strides=[2, 2, 2, 2],
        q_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    dist = DiscreteDistParams()

    # rl
    gamma: float = 0.99
    roll_length: int = 20
    reward_scaling: float = 1.0
    reward_clip_value: float = 10_000.0

    # training
    batch_size: int = 2000
    learning_rate: float = 0.002
    clip_norm: float = 1.0

    algo = SacParams(
        target_entropy_constant=1.7,
        init_alpha=0.2,
    )

    def get_fname_string_from_params(self):
        return f"LR_{self.learning_rate}_" + \
               f"R_{self.roll_length}_" + \
               f"BS_{self.batch_size}_" + \
               f"IA_{self.algo.init_alpha}_" + \
               f"TE_{self.algo.target_entropy_constant}_" + \
               f"G_{self.gamma}_" + \
               f"CN_{self.clip_norm}_" + \
               f"TEMP_{self.dist.temperature}_" + \
               f"HS_{self.arch.q_hidden_sizes}"

    def randomize(self, counter):
        self.dist.temperature = [1.0, 2.0, 3.0, 4.0][counter % 4]


if __name__ == "__main__":
    start_experiment_helper(
        "PONG_GUMBEL_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_discrete_vision_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )
