from dataclasses import dataclass
from functools import partial
from nis import cat
from typing import Tuple
from agency.algo.ppo.network import ConvNetworkArchitecture, PpoParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams

import torch
from agency.algo import ppo
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import TrainLoopParams, start_experiment_helper
from agency.layers.distributions import DiscreteDistParams


@dataclass
class WorldParams(GymWorldParams):
    name: str = "PongNoFrameskip-v4"
    env_class: str = "atari"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (2, 84, 84)
    render: bool = False
    num_workers: int = 200
    num_actions: int = 6
    use_envpool: bool = True

    def __post_init__(self):
        if self.use_envpool:
            self.name = "Pong-v5"


class HyperParams:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = LogParams(
        log_dir="/tmp/agency/logs/pong",
        train_samples_per_log=200_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=20_000,
    )

    data = DataCollectionParams(
        init_agent_steps=5_000,
        max_agent_steps=5_000_000,
    )

    arch = ConvNetworkArchitecture(
        shared_enc_channels=[8, 32, 64],
        shared_enc_kernel_sizes=[5, 3, 3],
        shared_enc_strides=[4, 2, 2],
        v_hidden_sizes=[512],
        p_hidden_sizes=[512],
    )

    dist = DiscreteDistParams(categorical=True)

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=20,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=1.0,
    )

    algo = PpoParams(
        p_learning_rate=0.0004,
        v_learning_rate=0.0004,
        ppo_clip=0.2,
        v_clip=0.2,
        clip_value_function=False,
        entropy_loss_scaling=0.01,
        normalize_advantage=False,
        use_dual_optimizer=False,
    )

    def get_fname_string_from_params(self):
        return (
            f"LRV_{self.algo.v_learning_rate}_"
            + f"LRP_{self.algo.p_learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"G_{self.rl.gamma}_"
            + f"CN_{self.backprop.clip_norm}_"
            + f"TEMP_{self.dist.temperature}_"
            + f"HS_{self.arch.v_hidden_sizes}"
        )

    def randomize(self, counter):
        self.dist.temperature = [1.0, 2.0, 3.0, 4.0][counter % 4]


if __name__ == "__main__":
    hp = HyperParams()
    start_experiment_helper(
        "PONG_CATEGORICAL_PPO",
        hp=hp,
        wp=WorldParams(),
        create_network_fn=partial(ppo.network.create_vision_network, normalize_input=False),
        create_train_data_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        create_memory_fn=ppo.memory.create_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
    )
