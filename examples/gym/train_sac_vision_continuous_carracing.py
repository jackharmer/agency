from dataclasses import dataclass
from typing import Tuple

import agency.algo.sac as sac
import torch
from agency.algo.sac.network import ContinuousDistParams, SacParams, VisionNetworkArchitecture
from agency.core.batch import create_batch
from agency.core.logger import LogParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import start_experiment_helper


@dataclass
class WorldParams:
    name: str = "CarRacing-v1"
    env_class: str = "box2d"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (84, 84, 3)
    render: bool = False
    episodes_per_render: int = 1
    num_workers: int = 10
    num_actions: int = 3
    use_vecenv: bool = True


class HyperParams:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log = LogParams(
        log_dir="/tmp/agency/logs/carracing",
        train_samples_per_log=100_000
    )

    # memory
    max_agent_steps: int = 1_000_000
    max_memory_size: int = 50_000
    init_steps: int = 2_000

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[16, 16, 32, 32],
        shared_enc_kernel_sizes=[3, 3, 3, 3],
        shared_enc_strides=[2, 2, 2, 2],
        q_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    dist = ContinuousDistParams()

    # rl
    gamma: float = 0.99
    roll_length: int = 10
    reward_scaling: float = 1.0
    reward_clip_value: float = 10.0

    # training
    batch_size: int = 2000
    learning_rate: float = 0.0005
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
        pass


if __name__ == "__main__":
    start_experiment_helper(
        "CARRACING_CONT_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_continuous_vision_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )
