from dataclasses import dataclass
from .episodic import EpisodicMemory


@dataclass
class MemoryParams:
    max_memory_size: int = 1_000_000
    is_circular: bool = True


def create_episodic_memory(hp, wp):
    return EpisodicMemory(max_size=hp.memory.max_memory_size, min_episode_length=hp.rl.roll_length)
