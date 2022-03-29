from dataclasses import dataclass
from typing import Tuple
from agency.algo.awac.network import AwacParams

from agency.algo import sac, awac
import numpy as np
import torch
from agency.algo.sac.network import (DiscreteDistParams, SacParams,
                                     VisionNetworkArchitecture)
from agency.core.batch import create_batch, create_batch_categorical
from agency.core.experiment import start_experiment
from agency.core.logger import LogParams
from agency.worlds.simulator import create_gym_simulator


@dataclass
class WorldParams:
    name: str = "MiniGrid-Empty-5x5-v0"
    env_class: str = "minigrid"
    is_image: bool = True
    input_size: Tuple[int, int, int] = (7, 7, 3)
    render: bool = False
    episodes_per_render: int = 3
    num_workers: int = 4
    num_actions: int = 7
    use_vecenv: bool = True


class BaseHyperParams:
    device = torch.device('cuda')

    log = LogParams(
        log_dir="/tmp/agency/tests/minigrid",
        train_samples_per_log=100_000
    )

    # memory
    max_agent_steps: int = 30_000
    max_memory_size: int = 200_000
    init_steps: int = 2_000

    arch = VisionNetworkArchitecture(
        shared_enc_channels=[16, 16, 32],
        shared_enc_kernel_sizes=[3, 3, 3],
        shared_enc_strides=[1, 1, 1],
        q_hidden_sizes=[512, 256],
        p_hidden_sizes=[512, 256],
    )

    # RL
    gamma: float = 0.99
    roll_length: int = 10
    reward_scaling: float = 1.0
    reward_clip_value: float = 10_000.0

    # TRAINING
    batch_size: int = 2000
    learning_rate: float = 0.002
    clip_norm: float = 1.0


class SacCategoricalHyperParams(BaseHyperParams):
    algo = SacParams(target_entropy_constant=0.4)
    dist = DiscreteDistParams(
        categorical=True
    )


class SacGumbelHyperParams(BaseHyperParams):
    algo = SacParams(
        target_entropy_constant=1.7,
    )
    dist = DiscreteDistParams(
        categorical=False,
    )


class AwacGumbelHyperParams(BaseHyperParams):
    algo = AwacParams(
        beta=1.0,
        adv_clip=1000,
        use_softmax=False,
    )
    dist = DiscreteDistParams(
        categorical=False,
        temperature=4
    )
    max_agent_steps: int = 60_000


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
        create_batch_fn=create_batch_categorical,
        train_on_batch_fn=sac.trainer.train_on_batch_categorical,
        create_simulator_fn=create_gym_simulator
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
        create_batch_fn=create_batch,
        train_on_batch_fn=sac.trainer.train_on_batch,
        create_simulator_fn=create_gym_simulator
    )

    infos = ep_info_buffer[-10:]
    mean_reward = np.mean([x.reward for x in infos])
    print(f"Mean reward {mean_reward}")

    assert len(infos) > 0
    assert mean_reward > 0.70


# TODO: This is too slow to train, tweak HPs.
# def test_awac_gumbel_vision_5x5():
#     setup()

#     ep_info_buffer = start_experiment(
#         hp=AwacGumbelHyperParams(),
#         wp=WorldParams(),
#         exp_name="AwacGumbelMiniGridVisionTest",
#         network_creater_fn=awac.network.create_discrete_vision_network,
#         train_data_creator_fn=awac.trainer.create_train_state_data,
#         create_inferer_fn=awac.trainer.create_inferer,
#         create_batch_fn=create_batch,
#         train_on_batch_fn=awac.trainer.train_on_batch,
#         create_simulator_fn=create_gym_simulator
#     )

#     infos = ep_info_buffer[-10:]
#     mean_reward = np.mean([x.reward for x in infos])
#     print(f"Mean reward {mean_reward}")

#     assert len(infos) > 0
#     assert mean_reward > 0.70
