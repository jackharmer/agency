import argparse
import math
import os
from functools import partial
from time import sleep
from typing import Any

import numpy as np
import torch
from agency.core.logger import Logger, LogParams
from agency.memory.episodic import EpisodicBuffer, EpisodicMemory


def print_header(title):
    print("""
        ___   _____________   __________  __
       /   | / ____/ ____/ | / / ____/\ \/ /
      / /| |/ / __/ __/ /  |/ / /      \  /
     / ___ / /_/ / /___/ /|  / /___    / /
    /_/  |_\____/_____/_/ |_/\____/   /_/
    """)
    print(title)


def get_unique_exp_path(directory, desired_name):
    full_path = os.path.join(directory, desired_name)
    if os.path.isdir(full_path):
        print(f"Path {full_path} exists looking for new path")
        counter = 1
        while True:
            new_path = full_path + f"_{counter}"
            if not os.path.isdir(new_path):
                full_path = new_path
                break
            counter += 1
    print(f"Found path {full_path}")
    return full_path


def collect_initial_data(memory, min_steps=0, min_episodes=1, sleep_time=1.0):
    mem_steps = memory.total_agent_steps()
    while mem_steps < min_steps or memory.num_complete_episodes() < min_episodes:
        sleep(sleep_time)
        mem_steps = memory.total_agent_steps()
        print(f"steps: {mem_steps}, episodes: {memory.num_complete_episodes()}")


def train_loop(
        net,
        simulator,
        memory,
        train_data,
        logger,
        episode_info_buffer,
        episode_info_output_buffer,  # permament store for episode infos
        create_batch_fn,
        train_on_batch_fn,
        max_train_steps,
        hp,
):
    cc = 0
    train_samples = 0
    while simulator.get_agent_steps() < hp.max_agent_steps:
        # with Timer(text="Mem samp: {:.4f}"):
        rolls = memory.sample(batch_size=hp.batch_size, roll_length=hp.roll_length)

        # with Timer(text="Train: {:.4f}"):
        mini_batch = create_batch_fn(rolls, hp.device)
        train_info = train_on_batch_fn(net, train_data, mini_batch)

        train_samples += train_info.samples

        episode_infos = episode_info_buffer.sample_all()
        if episode_info_output_buffer is not None:
            for info in episode_infos:
                episode_info_output_buffer.append(info)

        logger.update(
            agent_steps=simulator.get_agent_steps(),
            update_steps=cc,
            train_samples=train_samples,
            episode_infos=episode_infos,
            train_info=train_info
        )
        cc += 1
        if cc >= max_train_steps:
            print("TRAIN LOOP: Reached max train steps, exiting.")
            break


def start_training(
        *,
        net,
        simulator,
        memory,
        train_data,
        logger,
        episode_info_buffer,
        create_batch_fn,
        train_on_batch_fn,
        hp,
        max_train_steps=math.inf,
        debug=False,
        episode_info_output_buffer=None,  # permament store for episode infos
):
    if debug:
        from torch import autograd
        autograd.set_detect_anomaly(True)

    print("COLLECT INITIAL DATA")
    simulator.start()
    collect_initial_data(memory, min_steps=hp.init_steps, min_episodes=1)

    logger.start_timers()
    logger.log_hyper_params(hp)

    print("START TRAINING")
    train_loop(
        net,
        simulator,
        memory,
        train_data,
        logger,
        episode_info_buffer,
        episode_info_output_buffer,
        create_batch_fn,
        train_on_batch_fn,
        max_train_steps,
        hp
    )

    print("END TRAINING")
    simulator.stop()
    logger.stop()


def start_experiment(
    hp,
    wp,
    exp_name: str,
    network_creater_fn,
    train_data_creator_fn,
    create_inferer_fn,
    create_batch_fn,
    train_on_batch_fn,
    create_simulator_fn
):
    print_header(exp_name)
    print(hp.device)

    net = network_creater_fn(
        input_size=wp.input_size,
        arch=hp.arch,
        dist=hp.dist,
        algo=hp.algo,
        num_actions=wp.num_actions,
    ).to(hp.device)

    memory = EpisodicMemory(max_size=hp.max_memory_size, min_episode_length=hp.roll_length)
    episode_info_buffer = EpisodicBuffer()
    episode_info_output_buffer = EpisodicBuffer()

    train_data_container = train_data_creator_fn(net=net, hp=hp, wp=wp, categorical=hp.dist.categorical)
    # print("TARGET ENTROPY = ", train_data_container.target_entropy)

    start_training(
        net=net,
        simulator=create_simulator_fn(
            inferer=create_inferer_fn(net=net, wp=wp, device=hp.device),
            world_params=wp,
            memory=memory,
            episode_info_buffer=episode_info_buffer,
        ),
        memory=memory,
        train_data=train_data_container,
        logger=Logger(
            experiment_path=get_unique_exp_path(hp.log.log_dir, exp_name),
            agent_steps_per_log=hp.log.agent_steps_per_log,
            train_samples_per_log=hp.log.train_samples_per_log,
            log_on=hp.log.log_on
        ),
        episode_info_buffer=episode_info_buffer,
        create_batch_fn=create_batch_fn,
        train_on_batch_fn=train_on_batch_fn,
        hp=hp,
        episode_info_output_buffer=episode_info_output_buffer,
    )
    return episode_info_output_buffer


def launch_experiment_sweeps(
        param_sweep: bool,
        num_experiments: int,
        base_name: str,  # The base name for the experiment
        hp: Any,  # Hyper Parameters
        wp: Any,  # World Parameters
        start_experiment_fn: Any,
        torch_seed: int = None,
        np_seed: int = None,
):
    torch.set_num_threads(torch.get_num_threads())
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    if np_seed is not None:
        np.random.seed(np_seed)

    for N in range(num_experiments):
        if param_sweep:
            print(f"------- Parameter Sweep, Experiment: {N} -------")
            hp.randomize(N)
        start_experiment_fn(hp=hp, wp=wp, exp_name=base_name + "_" + hp.get_fname_string_from_params())


def start_experiment_helper(
    exp_base_name: str,
    hp,
    wp,
    create_network_fn,
    create_train_data_fn,
    create_inferer_fn,
    create_batch_fn,
    train_on_batch_fn,
    create_simulator_fn
):
    torch.set_printoptions(precision=8)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true', help="Run a parameter sweep.")
    parser.add_argument('--n', type=int, default=1, help="The number of sequential experiments to run.")
    args = parser.parse_args()

    launch_experiment_sweeps(
        args.sweep,
        args.n,
        exp_base_name,
        hp=hp,
        wp=wp,
        start_experiment_fn=partial(
            start_experiment,
            network_creater_fn=create_network_fn,
            train_data_creator_fn=create_train_data_fn,
            create_inferer_fn=create_inferer_fn,
            create_batch_fn=create_batch_fn,
            train_on_batch_fn=train_on_batch_fn,
            create_simulator_fn=create_simulator_fn
        )
    )
