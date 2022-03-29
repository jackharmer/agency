import argparse
from dataclasses import dataclass

import agency.algo.sac as sac
import numpy as np
import torch
from agency.algo.sac.network import (ContinuousDistParams,
                                     MlpNetworkArchitecture, SacData,
                                     SacParams)
from agency.core.batch import create_batch
from agency.core.experiment import (collect_initial_data, get_unique_exp_path,
                                    print_header, train_loop)
from agency.core.logger import Logger, LogParams
from agency.layers.distributions import GaussianPolicy
from agency.layers.feed_forward import mlp
from agency.layers.policy_heads import PolicyHead
from agency.layers.value_heads import QEncoder, QHead
from agency.memory.episodic import EpisodicBuffer, EpisodicMemory
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams:
    name: str = "LunarLanderContinuous-v2"
    env_class: str = "box2d"
    render: bool = False
    episodes_per_render: int = 1
    is_image: bool = False
    num_workers: int = 10
    input_size: int = 8 * 3
    num_actions: int = 2
    use_vecenv: bool = True


class HyperParams:
    device = torch.device('cuda')

    log = LogParams(
        log_dir="/tmp/agency/logs/lunar_lander",
        train_samples_per_log=200_000
    )

    # memory
    max_agent_steps: int = 500_000
    max_memory_size: int = 1_000_000
    init_steps: int = 20_000

    arch = MlpNetworkArchitecture(
        q_hidden_sizes=[256, 256],
        p_hidden_sizes=[256, 256]
    )

    dist = ContinuousDistParams()

    # rl
    gamma: float = 0.99
    roll_length: int = 10
    reward_scaling: float = 1.0
    reward_clip_value: float = 10_000.0

    # training
    batch_size: int = 2000
    learning_rate: float = 0.008
    clip_norm: float = 1.0

    algo = SacParams(
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
        GaussianPolicy(p_hidden[-1], num_actions)
    )

    return SacData(
        q1=QHead(q1_enc, input_size=q_hidden[-1], output_size=1),
        q2=QHead(q2_enc, input_size=q_hidden[-1], output_size=1),
        q1_target=QHead(q1_target_enc, input_size=q_hidden[-1], output_size=1),
        q2_target=QHead(q2_target_enc, input_size=q_hidden[-1], output_size=1),
        policy=policy,
        log_alpha=torch.tensor(np.log(algo.init_alpha), dtype=torch.float32, requires_grad=True),
        policy_params=list(policy.parameters())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sac_tutorial', help="experiment name")
    args = parser.parse_args()

    print_header(args.name)

    hp = HyperParams()
    wp = WorldParams()

    net = create_network(
        input_size=wp.input_size,
        arch=hp.arch,
        dist=hp.dist,
        algo=hp.algo,
        num_actions=wp.num_actions
    )
    net.to(hp.device)

    memory = EpisodicMemory(max_size=hp.max_memory_size, min_episode_length=hp.roll_length)
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
        log_on=hp.log.log_on
    )

    simulator.start()
    collect_initial_data(memory, min_steps=hp.init_steps, min_episodes=1)

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
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        max_train_steps=1_000_000,
        hp=hp
    )

    simulator.stop()
    logger.stop()
