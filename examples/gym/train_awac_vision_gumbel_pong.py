from dataclasses import dataclass
from typing import Tuple

from agency.algo.awac.network import AwacParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams

import torch
from agency.algo import awac, sac
from agency.algo.sac.network import DiscreteDistParams, VisionNetworkArchitecture
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import TrainLoopParams, start_experiment_helper

import gym


@dataclass
class WorldParams(GymWorldParams):
    name: str = "PongNoFrameskip-v4"
    env_class: str = "atari"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (2, 84, 84)
    render: bool = False
    num_workers: int = 100
    num_actions: int = 6
    use_envpool: bool = False
    frame_stack: int = 2

    def __post_init__(self):
        if self.use_envpool:
            self.name = "Pong-v5"


class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/pong",
        train_samples_per_log=100_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=20_000,
    )

    data = DataCollectionParams(
        init_agent_steps=5_000,
        max_agent_steps=5_000_000,
    )

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[8, 32, 64],
        shared_enc_kernel_sizes=[5, 3, 3],
        shared_enc_strides=[4, 2, 2],
        q_hidden_sizes=[512],
        p_hidden_sizes=[512],
    )

    dist = DiscreteDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=20,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=0.5,
    )

    algo = AwacParams(learning_rate=0.001)

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"G_{self.rl.gamma}_"
            + f"CN_{self.backprop.clip_norm}_"
            + f"TEMP_{self.dist.temperature}_"
            + f"HS_{self.arch.q_hidden_sizes}"
        )

    def randomize(self, counter):
        self.dist.temperature = [1.0, 2.0, 3.0, 4.0][counter % 4]


if __name__ == "__main__":
    # This takes a long time to train. Hyper parameters need tuning.
    start_experiment_helper(
        "PONG_GUMBEL_AWAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=awac.network.create_discrete_vision_network,
        create_train_data_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
    )
