import random
from dataclasses import dataclass

import torch
import agency.algo.ppo as ppo
from agency.algo.ppo.network import MlpNetworkArchitecture, PpoParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.logger import LogParams
from agency.layers.distributions import DiscreteDistParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator
from agency.core.experiment import start_experiment_helper, TrainLoopParams


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
        train_samples_per_log=100_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=5_000,
    )

    data = DataCollectionParams(
        init_agent_steps=1_000,
        init_episodes=1,
        max_agent_steps=500_000,
    )

    arch = MlpNetworkArchitecture(
        shared_hidden_sizes=None,
        v_hidden_sizes=[25, 25],
        p_hidden_sizes=[25, 25],
    )

    dist = DiscreteDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=1,
    )

    backprop = BackpropParams(
        batch_size=1000,
        clip_norm=10.0,
    )

    algo = PpoParams(
        p_learning_rate=0.0002,
        v_learning_rate=0.0002,
        ppo_clip=0.2,
        clip_value_function=False,
        entropy_loss_scaling=0.001,
        normalize_advantage=False,
    )

    def get_fname_string_from_params(self):
        return (
            f"LRP_{self.algo.p_learning_rate}_"
            + f"LRV_{self.algo.v_learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"CLIP_{self.algo.ppo_clip}_"
            + f"CLIPVF_{self.algo.clip_value_function}_"
            + f"NA_{self.algo.normalize_advantage}_"
            + f"G_{self.rl.gamma}_"
            + f"CN_{self.backprop.clip_norm}_"
            + f"HS_{self.arch.v_hidden_sizes}"
        )

    def randomize(self, counter):
        self.algo.p_learning_rate = random.choice([0.001, 0.003])
        self.algo.v_learning_rate = random.choice([0.001, 0.003])
        self.algo.init_alpha = random.choice([0.1])
        self.algo.target_entropy_constant = random.choice([1.0])


if __name__ == "__main__":
    start_experiment_helper(
        "CARTPOLE_CONT_PPO",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=ppo.network.create_network,
        create_train_data_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        create_memory_fn=ppo.memory.create_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
    )
