from dataclasses import dataclass

import agency.algo.ppo as ppo
import torch
from agency.algo.ppo.batch import create_batch
from agency.algo.ppo.network import MlpNetworkArchitecture, PpoParams
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
    episodes_per_render: int = 2
    num_workers: int = 100
    input_size: int = 8 * 3
    num_actions: int = 2


class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/lunar_lander",
        train_samples_per_log=400_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=5_000,
    )

    data = DataCollectionParams(
        init_agent_steps=5_000,
        max_agent_steps=5_000_000,
    )

    arch = MlpNetworkArchitecture(
        # Use seperate columns:
        # shared_hidden_sizes=None,
        # v_hidden_sizes=[256, 256],
        # p_hidden_sizes=[256, 256]
        # Use shared layers before output layer:
        shared_hidden_sizes=[256, 256],
        v_hidden_sizes=None,
        p_hidden_sizes=None,
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

    use_prealloc_memory: bool = True

    algo = PpoParams(
        p_learning_rate=0.0002,
        v_learning_rate=0.0002,
        ppo_clip=0.2,
        clip_value_function=False,
        entropy_loss_scaling=0.001,
        normalize_advantage=False,
        use_terminal_masking=True,
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
        pass


if __name__ == "__main__":
    torch.set_printoptions(precision=3)
    hp = HyperParams()
    start_experiment_helper(
        "LL_CONT_PPO",
        hp=hp,
        wp=WorldParams(),
        create_network_fn=ppo.network.create_network,
        create_train_data_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=ppo.batch.create_batch_from_block_memory if hp.use_prealloc_memory else create_batch,
        create_memory_fn=ppo.memory.create_block_memory
        if hp.use_prealloc_memory
        else create_episodic_memory,
    )
