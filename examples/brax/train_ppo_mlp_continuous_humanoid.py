import os
from functools import partial
from time import sleep

# GPU memory. By default JAX will pre-allocate 90% of the available GPU memory:
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from dataclasses import dataclass

import torch
import torch.nn as nn

import agency.algo.ppo as ppo
from agency.algo.ppo.network import MlpNetworkArchitecture, PpoParams
from agency.core import BackpropParams, DataCollectionParams, GenericRlParams
from agency.core.experiment import TrainLoopParams, start_experiment_helper
from agency.core.logger import LogParams
from agency.layers.distributions import ContinuousDistParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_brax_gym_simulator

roll_length = 32
num_workers = 4096


@dataclass
class HumanoidWorldParams(GymWorldParams):
    name: str = "humanoid"
    env_class: str = "brax"
    render: bool = True
    episodes_per_render: int = 1
    num_workers: int = num_workers
    input_size: int = 338
    num_actions: int = 17
    obs_clamp: float = 5


@dataclass
class HyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/logs/brax_humanoid",
        train_samples_per_log=700_000,
        use_wandb=False,
        wandb_project_name="agency_brax_humanoid",
    )

    train = TrainLoopParams(post_backprop_sleep=0.01)

    memory = MemoryParams(
        max_memory_size=roll_length * num_workers,
        is_circular=True,
    )

    data = DataCollectionParams(
        init_agent_steps=roll_length * num_workers,
        max_agent_steps=5_000_000_000,
    )

    arch = MlpNetworkArchitecture(
        shared_hidden_sizes=None,
        v_hidden_sizes=[512, 256, 128],
        p_hidden_sizes=[512, 256, 128],
    )

    dist = ContinuousDistParams()

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=roll_length,
        reward_scaling=1.00,
        reward_clip_value=5000.0,
    )

    backprop = BackpropParams(
        batch_size=1024 * 32,
        clip_norm=1.0,
    )

    algo = PpoParams(
        p_learning_rate=0.0003,
        v_learning_rate=0.0003,
        ppo_clip=0.3,
        v_clip=0.2,
        clip_value_function=False,
        entropy_loss_scaling=1e-6,
        v_loss_scaling=1,
        normalize_advantage=True,
        normalize_value=True,
        use_gae=True,
        gae_lambda=0.95,
        use_dual_optimizer=False,
        use_terminal_masking=True,
        use_state_independent_std=True,
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
    hp,
):
    blocking_memory = not memory.is_circular
    if blocking_memory:
        num_batches = max(1, memory.capacity() // hp.backprop.batch_size)
        mini_epochs = 5
    else:
        num_batches = 1
        mini_epochs = 1

    update_step = 0
    train_samples = 0
    while simulator.get_agent_steps() < hp.data.max_agent_steps:
        while not memory.is_full():
            sleep(0.0001)

        if blocking_memory:
            rolls = memory.sample_all(chunks=num_batches)
        else:
            rolls = [
                memory.sample(batch_size=hp.backprop.batch_size, roll_length=hp.rl.roll_length)
                for _ in range(num_batches)
            ]

        if blocking_memory:
            memory.clear()

        mini_batches = [create_batch_fn(roll, hp.device) for roll in rolls]

        train_data.value_normalizer.training = True
        train_infos = []
        for _ in range(mini_epochs):
            for mini_batch in mini_batches:
                train_info = train_on_batch_fn(hp, net, train_data, mini_batch, logger.should_log())
                train_infos.append(train_info)
            train_data.value_normalizer.training = False

        # This is useful for slowing down the backprops in order to give more compute resources
        # to the environments.
        if hp.train.post_backprop_sleep > 0:
            sleep(hp.train.post_backprop_sleep)

        for train_info in train_infos:
            train_samples += train_info.samples

        episode_infos = episode_info_buffer.sample_all()
        if episode_info_output_buffer is not None:
            episode_info_output_buffer.extend(episode_infos)

        logger.update(
            agent_steps=simulator.get_agent_steps(),
            update_steps=update_step,
            train_samples=train_samples,
            episode_infos=episode_infos,
            train_info=train_info,  # TODO: add functionality to pass a list of train_info
        )
        update_step += 1

        if update_step >= hp.train.max_backprop_steps:
            print("TRAIN LOOP: Reached max train steps, exiting.")
            break


if __name__ == "__main__":
    hp = HyperParams()
    wp = HumanoidWorldParams()

    start_experiment_helper(
        "BRAX_HUMANOID_CONT_PPO",
        hp=hp,
        wp=wp,
        create_network_fn=partial(ppo.network.create_network, activation=nn.ELU),
        create_train_data_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_brax_gym_simulator,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        create_memory_fn=ppo.memory.create_block_memory,
        train_loop_fn=train_loop,
        seed=7,
    )
