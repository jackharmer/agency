from agency.algo.sac.batch import SacBlockOfSteps
from agency.memory.block_memory import BlockOfStepsMemory


def create_block_memory(hp, wp):
    max_size = hp.memory.max_memory_size // wp.num_workers
    sd = SacBlockOfSteps(
        wp.num_workers,
        max_size,
        wp.input_size,
        wp.num_actions,
        hp.device,
        hp.dist.categorical,
    )
    return BlockOfStepsMemory(sd, device=hp.device)
