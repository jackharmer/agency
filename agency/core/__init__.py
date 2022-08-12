from dataclasses import dataclass
import math


@dataclass
class GenericRlParams:
    gamma: float = 0.99
    roll_length: int = 10
    reward_scaling: float = 1.0
    reward_clip_value: float = 10000.0


@dataclass
class BackpropParams:
    batch_size: int = 2000
    clip_norm: float = 1.0


@dataclass
class DataCollectionParams:
    init_agent_steps: int = 20_000
    init_episodes: int = 0
    max_agent_steps: int = math.inf
