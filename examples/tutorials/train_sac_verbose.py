import argparse
from dataclasses import dataclass

import agency.algo.sac as sac
import numpy as np
import torch
from agency.algo.sac.network import MlpNetworkArchitecture, SacNetwork, SacParams
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.experiment import (
    TrainLoopParams,
    collect_initial_data,
    get_unique_exp_path,
    print_header,
    train_loop,
)
from agency.core.logger import Logger, LogParams
from agency.layers.distributions import GaussianPolicy, ContinuousDistParams
from agency.layers.feed_forward import mlp
from agency.layers.policy_heads import PolicyHead
from agency.layers.value_heads import QEncoder, QHead
from agency.memory import MemoryParams
from agency.memory.episodic import EpisodicBuffer, EpisodicMemory
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
        train_samples_per_log=400_000,
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


def create_network(
    input_size,
    arch,
    dist,
    algo,
    num_actions,
):
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
    hp = HyperParams()
    wp = WorldParams()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="sac_tutorial", help="experiment name")
    parser.add_argument("--max_backprops", type=int, default=hp.train.max_backprop_steps)
    args = parser.parse_args()
    hp.train.max_backprop_steps = args.max_backprops

    print_header(args.name)

    net = create_network(
        input_size=wp.input_size,
        arch=hp.arch,
        dist=hp.dist,
        algo=hp.algo,
        num_actions=wp.num_actions,
    )
    net.to(hp.device)

    memory = sac.memory.create_block_memory(hp, wp)
    episode_info_buffer = EpisodicBuffer()

    simulator = create_gym_simulator(
        inferer=sac.trainer.create_inferer(net=net, wp=wp, device=hp.device),
        world_params=wp,
        memory=memory,
        episode_info_buffer=episode_info_buffer,
    )

    logger = Logger(
        experiment_path=get_unique_exp_path(hp.log.log_dir, args.name),
        agent_steps_per_log=hp.log.agent_steps_per_log,
        train_samples_per_log=hp.log.train_samples_per_log,
        log_on=hp.log.log_on,
    )

    simulator.start()
    collect_initial_data(memory, min_steps=hp.data.init_agent_steps, min_episodes=1)

    logger.start_timers()
    logger.log_hyper_params(hp)

    train_loop(
        net=net,
        simulator=simulator,
        memory=memory,
        train_data=sac.trainer.create_train_state_data(net=net, hp=hp, wp=wp),
        logger=logger,
        episode_info_buffer=episode_info_buffer,
        episode_info_output_buffer=None,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch,
        hp=hp,
    )

    simulator.stop()
    logger.stop()
