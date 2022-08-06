import time
from typing import Any

from agency.worlds.gym_env import (
    GymThread,
    GymVecEnvThread,
    gym_create_envpool_vecenv,
    gym_create_single_env,
    gym_create_async_vecenv,
)
from agency.worlds.unity_env import UnityThread


class AsyncSimulator:
    def __init__(
        self,
        inferer: Any,
        num_worker_threads: int,
        world_params: Any,
        memory: Any,
        episode_info_buffer: Any,
        thread_class: Any,
        make_env_fn: Any,
    ):
        self._inferer = inferer
        self._num_workers_threads = num_worker_threads
        self._params = world_params
        self._memory = memory
        self._episode_info_buffer = episode_info_buffer
        self._world_threads = []
        for thread_counter in range(self._num_workers_threads):
            self._world_threads.append(
                thread_class(
                    thread_id=thread_counter,
                    inferer=self._inferer,
                    memory=self._memory,
                    params=self._params,
                    episode_info_buffer=self._episode_info_buffer,
                    make_env_fn=make_env_fn,
                )
            )

    def start(self):
        for thread in self._world_threads:
            thread.start()

    def stop(self):
        print("SIMULATOR: ENVS STOP REQUESTED")
        for thread in self._world_threads:
            thread.request_stop()

        print("SIMULATOR: ENVS WAITING")
        all_finished = False
        while not all_finished:
            all_finished = True
            for thread in self._world_threads:
                ready_to_stop = thread.is_ready_to_stop()
                all_finished = all_finished and ready_to_stop
                if not ready_to_stop:
                    time.sleep(0.01)
        print("SIMULATOR: ENVS STOPPED")

        # print("SIMULATOR: THREADS STOP REQUESTED")
        # for thread in self._world_threads:
        #     thread.stop()

        print("SIMULATOR: ENVS JOINING")
        for thread in self._world_threads:
            # thread.terminate()
            thread.join()

        print("All threads finished")

    def get_agent_steps(self):
        return self._memory.total_agent_steps()

    def is_synchronous(self):
        return False


def create_gym_simulator(
    inferer,
    world_params,
    memory,
    episode_info_buffer,
):
    if world_params.use_vecenv:
        num_worker_threads = 1  # vecenvs launch the seperate world instances.
        make_env_fn = gym_create_envpool_vecenv if world_params.use_envpool else gym_create_async_vecenv
        thread_class = GymVecEnvThread
    else:
        num_worker_threads = world_params.num_workers
        make_env_fn = gym_create_single_env
        thread_class = GymThread

    simulator = AsyncSimulator(
        inferer=inferer,
        num_worker_threads=num_worker_threads,
        world_params=world_params,
        memory=memory,
        episode_info_buffer=episode_info_buffer,
        thread_class=thread_class,
        make_env_fn=make_env_fn,
    )

    return simulator


def create_unity_simulator(inferer, world_params, memory, episode_info_buffer, **kwargs):
    simulator = AsyncSimulator(
        inferer=inferer,
        num_worker_threads=world_params.num_workers,
        world_params=world_params,
        memory=memory,
        episode_info_buffer=episode_info_buffer,
        thread_class=UnityThread,
        make_env_fn=None,
    )
    return simulator
