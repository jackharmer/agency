from dataclasses import dataclass

import agency.algo.sac as sac
import numpy as np
import torch
from agency.algo.sac.network import MlpNetworkArchitecture, SacNetwork, SacParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.experiment import TrainLoopParams, start_experiment_helper
from agency.core.logger import LogParams
from agency.layers.distributions import GaussianPolicy, ContinuousDistParams
from agency.layers.feed_forward import mlp
from agency.layers.policy_heads import PolicyHead
from agency.layers.value_heads import QEncoder, QHead
from agency.memory import MemoryParams
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
        init_agent_steps=20_000,
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
        learning_rate=0.008,
        init_alpha=0.1,
    )

    def get_fname_string_from_params(self):
        return (
            f"LR_{self.algo.learning_rate}_"
            + f"R_{self.rl.roll_length}_"
            + f"BS_{self.backprop.batch_size}_"
            + f"IA_{self.algo.init_alpha}_"
            + f"TE_{self.algo.target_entropy_constant}_"
            + f"G_{self.rl.gamma}"
        )

    def randomize(self, counter):
        pass


def create_network(input_size, arch, dist, algo, num_actions):
    q_input_size = input_size + num_actions
    q_hidden = arch.q_hidden_sizes
    p_input_size = input_size
    p_hidden = arch.p_hidden_sizes

    q1_enc = QEncoder(mlp(q_input_size, q_hidden))
    q2_enc = QEncoder(mlp(q_input_size, q_hidden))

    q1_target_enc = QEncoder(mlp(q_input_size, q_hidden))
    q2_target_enc = QEncoder(mlp(q_input_size, q_hidden))

    policy = PolicyHead(
        mlp(p_input_size, p_hidden),
        GaussianPolicy(p_hidden[-1], num_actions),
    )

    return SacNetwork(
        q1=QHead(q1_enc, input_size=q_hidden[-1], output_size=1),
        q2=QHead(q2_enc, input_size=q_hidden[-1], output_size=1),
        q1_target=QHead(q1_target_enc, input_size=q_hidden[-1], output_size=1),
        q2_target=QHead(q2_target_enc, input_size=q_hidden[-1], output_size=1),
        policy=policy,
        log_alpha=torch.tensor(np.log(algo.init_alpha), dtype=torch.float32, requires_grad=True),
        policy_params=list(policy.parameters()),
    )


if __name__ == "__main__":
    start_experiment_helper(
        "sac_tutorial",
        hp=HyperParams(),
        wp=WorldParams(),
        create_network_fn=create_network,
        create_train_data_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        create_memory_fn=sac.memory.create_block_memory,
    )
