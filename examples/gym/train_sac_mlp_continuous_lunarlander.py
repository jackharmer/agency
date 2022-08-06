import random
from dataclasses import dataclass

import torch
from agency.algo import sac
from agency.algo.sac.network import MlpNetworkArchitecture, SacParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.layers.distributions import ContinuousDistParams
from agency.algo.sac.batch import create_batch
from agency.core.logger import LogParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import TrainLoopParams, start_experiment_helper


@dataclass
class WorldParams(GymWorldParams):
    name: str = "LunarLanderContinuous-v2"
    env_class: str = "box2d"
    render: bool = False
    num_workers: int = 100
    input_size: int = 8 * 3
    num_actions: int = 2


class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/lunar_lander",
        train_samples_per_log=200_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=100_000,
    )

    data = DataCollectionParams(
        init_agent_steps=5_000,
        init_episodes=1,
        max_agent_steps=500_000,
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
        batch_size=2000,
        clip_norm=1.0,
    )

    algo = SacParams(
        learning_rate=0.007,
        init_alpha=0.1,
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
            + f"HS_{self.arch.q_hidden_sizes}"
        )

    def randomize(self, counter):
        self.algo.learning_rate = random.choice([0.001, 0.003, 0.008])
        self.algo.init_alpha = random.choice([0.1])
        self.algo.target_entropy_constant = random.choice([1.0])


if __name__ == "__main__":
    start_experiment_helper(
        "LL_CONT_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_continuous_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
    )
