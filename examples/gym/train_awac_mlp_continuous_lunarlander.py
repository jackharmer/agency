import random
from dataclasses import dataclass

import torch
from agency.algo import awac, sac
from agency.algo.awac.network import AwacParams
from agency.algo.sac.batch import create_batch
from agency.algo.sac.network import MlpNetworkArchitecture
from agency.core import BackpropParams, DataCollectionParams, GenericRlParams
from agency.core.experiment import TrainLoopParams, start_experiment_helper
from agency.core.logger import LogParams
from agency.layers.distributions import ContinuousDistParams
from agency.memory import MemoryParams, create_episodic_memory
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams(GymWorldParams):
    name: str = "LunarLanderContinuous-v2"
    env_class: str = "box2d"
    render: bool = False
    num_workers: int = 100
    input_size: int = 8 * 3
    num_actions: int = 2
    frame_stack: int = 3


class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/lunar_lander",
        train_samples_per_log=200_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=1_000_000,
    )

    data = DataCollectionParams(
        init_agent_steps=20_000,
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

    algo = AwacParams(learning_rate=0.008)

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"B_{self.algo.beta}_"
            + f"AC_{self.algo.adv_clip}_"
            + f"G_{self.rl.gamma}_"
            + f"SOFT_{self.algo.use_softmax}_"
            + f"CN_{self.backprop.clip_norm}"
        )

    def randomize(self, counter):
        self.algo.learning_rate = random.choice([0.0001, 0.0005, 0.001, 0.003])
        self.rl.roll_length = random.choice([5, 10, 20])
        self.backprop.batch_size = random.choice([500, 1000, 2000, 4000])
        self.algo.beta = random.choice([0.1, 0.2, 0.3])
        self.algo.adv_clip = random.choice([1, 2])
        self.algo.use_softmax = random.choice([False, True])


if __name__ == "__main__":
    start_experiment_helper(
        "LL_CONT_AWAC",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=awac.network.create_continuous_network,
        create_train_data_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )
