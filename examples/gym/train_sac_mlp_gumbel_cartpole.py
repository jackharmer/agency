import random
from dataclasses import dataclass

import torch
import agency.algo.sac as sac
from agency.algo.sac.network import MlpNetworkArchitecture, SacParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
from agency.layers.distributions import DiscreteDistParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import TrainLoopParams, start_experiment_helper


@dataclass
class WorldParams(GymWorldParams):
    name: str = "CartPole-v1"
    env_class: str = "box2d"
    render: bool = True
    episodes_per_render: int = 3
    num_workers: int = 10
    input_size: int = 4 * 3
    num_actions: int = 2


class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/cartpole",
        train_samples_per_log=10_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000_000,
    )

    data = DataCollectionParams(
        init_agent_steps=100,
        init_episodes=1,
        max_agent_steps=50_000,
    )

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[25, 25],
        p_hidden_sizes=[25, 25],
    )

    dist = DiscreteDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=1,
    )

    backprop = BackpropParams(
        batch_size=100,
        clip_norm=10.0,
    )

    algo = SacParams(learning_rate=0.003)

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
        self.algo.learning_rate = random.choice([0.001, 0.003])
        self.algo.init_alpha = random.choice([0.1])
        self.algo.target_entropy_constant = random.choice([1.0])


if __name__ == "__main__":
    start_experiment_helper(
        "CARTPOLE_CONT_SAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=sac.network.create_discrete_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
    )
