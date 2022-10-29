from dataclasses import dataclass
from typing import Tuple
from agency.algo.awac.network import AwacParams

from agency.algo import sac, ppo, awac
import numpy as np
import torch
from agency.algo.ppo.network import PpoParams
from agency.algo.sac.network import DiscreteDistParams, SacParams, VisionNetworkArchitecture
from agency.core import BackpropParams, GenericRlParams, DataCollectionParams
from agency.core.experiment import TrainLoopParams, start_experiment
from agency.core.logger import LogParams
from agency.memory import MemoryParams
from agency.worlds.gym_env import GymWorldParams
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams(GymWorldParams):
    name: str = "MiniGrid-Empty-5x5-v0"
    env_class: str = "minigrid"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (3, 7, 7)
    render: bool = False
    episodes_per_render: int = 3
    num_workers: int = 40
    num_actions: int = 7


class BaseHyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/tests/minigrid",
        train_samples_per_log=100_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=200_000,
    )

    data = DataCollectionParams(
        init_agent_steps=2_000,
        max_agent_steps=100_000,
    )

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[16, 16, 32],
        shared_enc_kernel_sizes=[3, 3, 3],
        shared_enc_strides=[1, 1, 1],
        q_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=10,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=1.0,
    )


class SacCategoricalHyperParams(BaseHyperParams):
    algo = SacParams(
        learning_rate=0.002,
        target_entropy_constant=0.4,
    )
    dist = DiscreteDistParams(categorical=True)


class SacGumbelHyperParams(BaseHyperParams):
    algo = SacParams(
        learning_rate=0.001,
        target_entropy_constant=1.7,
    )
    dist = DiscreteDistParams(
        categorical=False,
    )
    data = DataCollectionParams(
        init_agent_steps=2_000,
        max_agent_steps=100_000,
    )


class AwacGumbelHyperParams(BaseHyperParams):
    algo = AwacParams(
        learning_rate=0.001,
        beta=0.2,
        adv_clip=2,
        use_softmax=False,
    )
    dist = DiscreteDistParams(categorical=False, temperature=4)
    data = DataCollectionParams(
        init_agent_steps=2_000,
        max_agent_steps=100_000,
    )


class PpoGumbelHyperParams:
    device = torch.device("cuda")

    log = LogParams(
        log_dir="/tmp/agency/tests/minigrid",
        train_samples_per_log=100_000,
    )

    train = TrainLoopParams()

    memory = MemoryParams(
        max_memory_size=5_000,
    )

    data = DataCollectionParams(
        init_agent_steps=2_000,
        max_agent_steps=60_000,
    )

    arch = ppo.network.ConvNetworkArchitecture(
        shared_enc_channels=[16, 16, 32],
        shared_enc_kernel_sizes=[3, 3, 3],
        shared_enc_strides=[1, 1, 1],
        v_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    rl = GenericRlParams(
        gamma=0.99,
        roll_length=10,
    )

    backprop = BackpropParams(
        batch_size=2000,
        clip_norm=1.0,
    )

    algo = PpoParams(
        p_learning_rate=0.0002,
        v_learning_rate=0.0002,
        ppo_clip=0.2,
        v_clip=0.2,
        clip_value_function=False,
        entropy_loss_scaling=0.001,
        normalize_advantage=False,
        use_dual_optimizer=False,
    )
    dist = DiscreteDistParams(categorical=False)


class SacGumbelHyperParams(BaseHyperParams):
    algo = SacParams(
        learning_rate=0.002,
        target_entropy_constant=1.7,
    )
    dist = DiscreteDistParams(
        categorical=False,
    )


def setup():
    torch.set_num_threads(torch.get_num_threads())
    torch.manual_seed(42)
    np.random.seed(42)


def test_sac_categorical_vision_5x5():
    setup()

    ep_info_buffer = start_experiment(
        hp=SacCategoricalHyperParams(),
        wp=WorldParams(),
        exp_name="SacCategoricalMiniGridVisionTest",
        network_creater_fn=sac.network.create_discrete_vision_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_categorical_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch_categorical,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 0.70


def test_sac_gumbel_vision_5x5():
    setup()

    ep_info_buffer = start_experiment(
        hp=SacGumbelHyperParams(),
        wp=WorldParams(),
        exp_name="SacGumbelMiniGridVisionTest",
        network_creater_fn=sac.network.create_discrete_vision_network,
        train_data_creator_fn=sac.trainer.create_train_state_data,
        create_inferer_fn=sac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 0.70


def test_ppo_gumbel_vision_5x5():
    setup()

    ep_info_buffer = start_experiment(
        hp=PpoGumbelHyperParams(),
        wp=WorldParams(),
        exp_name="PpoGumbelMiniGridVisionTest",
        network_creater_fn=ppo.network.create_vision_network,
        train_data_creator_fn=ppo.trainer.create_train_state_data,
        create_inferer_fn=ppo.trainer.create_inferer,
        create_batch_fn=ppo.batch.create_batch_from_block_memory,
        train_on_batch_fn=ppo.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=ppo.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 0.70


def test_awac_gumbel_vision_5x5():
    setup()

    ep_info_buffer = start_experiment(
        hp=AwacGumbelHyperParams(),
        wp=WorldParams(),
        exp_name="AwacGumbelMiniGridVisionTest",
        network_creater_fn=awac.network.create_discrete_vision_network,
        train_data_creator_fn=awac.trainer.create_train_state_data,
        create_inferer_fn=awac.trainer.create_inferer,
        create_batch_fn=sac.batch.create_batch_from_block_memory,
        train_on_batch_fn=awac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator,
        create_memory_fn=sac.memory.create_block_memory,
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 0.70
