from dataclasses import dataclass
from typing import Tuple

import agency.algo.sac as sac
import torch
from agency.algo.sac.network import SacParams, VisionNetworkArchitecture
from agency.core import BackpropParams, DataCollectionParams, GenericRlParams
from agency.layers.distributions import ContinuousDistParams
from agency.algo.sac.batch import create_batch
from agency.core.logger import LogParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import start_experiment_helper, TrainLoopParams


@dataclass
class WorldParams(GymWorldParams):
    name: str = "CarRacing-v1"
    env_class: str = "box2d"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (3, 84, 84)
    render: bool = True
    num_workers: int = 40
    num_actions: int = 3


class HyperParams:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = LogParams(
        log_dir="/tmp/agency/logs/carracing",
        train_samples_per_log=100_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=20_000,
    )

    data = DataCollectionParams(
        init_agent_steps=2_000,
        init_episodes=0,
        max_agent_steps=1_000_000,
    )

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[16, 16, 32, 32],
        shared_enc_kernel_sizes=[3, 3, 3, 3],
        shared_enc_strides=[2, 2, 2, 2],
        q_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    dist = ContinuousDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=10,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=1.0,
    )

    algo = SacParams(learning_rate=0.0005)

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"IA_{self.algo.init_alpha}_"
            + f"TE_{self.algo.target_entropy_constant}_"
            + f"G_{self.rl.gamma}_"
            + f"CN_{self.backprop.clip_norm}_"
            + f"HS_{self.arch.q_hidden_sizes}"
        )

    def randomize(self, counter):
        pass


if __name__ == "__main__":
    start_experiment_helper(
        "CARRACING_CONT_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_continuous_vision_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
    )
