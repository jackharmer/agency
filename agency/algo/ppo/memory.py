from agency.algo.ppo.batch import PpoBlockOfSteps
from agency.memory.block_memory import BlockOfStepsMemory


def create_block_memory(hp, wp):
    max_size = hp.memory.max_memory_size // wp.num_workers
    sd = PpoBlockOfSteps(wp.num_workers, max_size, wp.input_size, wp.num_actions, hp.device)
    return BlockOfStepsMemory(sd, device=hp.device)
