from functools import partial
import random
from dataclasses import dataclass
from typing import Tuple

from agency.algo import sac
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
import torch
from agency.algo.sac.network import (
    DiscreteDistParams,
    SacParams,
    VisionNetworkArchitecture,
)
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import start_experiment_helper, TrainLoopParams


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

    def __post_init__(self):
        if self.use_envpool:
            self.name = "Pong-v5"


class HyperParams:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        max_agent_steps=1_000_000,
    )

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[8, 32, 64],
        shared_enc_kernel_sizes=[5, 3, 3],
        shared_enc_strides=[4, 2, 2],
        q_hidden_sizes=[512],
        p_hidden_sizes=[512],
    )

    dist = DiscreteDistParams(categorical=True)

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=20,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=0.5,
    )

    algo = SacParams(
        learning_rate=0.001,
        target_entropy_constant=0.2,
    )

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"IA_{self.algo.init_alpha}_"
            + f"TE_{self.algo.target_entropy_constant}_"
            + f"G_{self.rl.gamma}_"
            + f"CN_{self.backprop.clip_norm}_"
            + f"HS_{self.arch.q_hidden_sizes}_"
        )

    def randomize(self, counter):
        self.algo.target_entropy_constant = random.choice([0.1, 0.2, 0.3, 0.15, 0.25, 0.35])


if __name__ == "__main__":
    start_experiment_helper(
        "PONG_CAT_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=partial(sac.network.create_discrete_vision_network, normalize_input=False),
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        train_on_batch_fn=sac.trainer.train_on_batch_categorical,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=sac.batch.create_categorical_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
    )
