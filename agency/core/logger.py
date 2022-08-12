from dataclasses import dataclass
from typing import Any
from enum import Enum
from agency.tools.timer import TimeDelta
import torch
from torch.utils.tensorboard import SummaryWriter


class LogOn(Enum):
    TRAIN_SAMPLES = 1
    AGENT_STEPS = 2


@dataclass
class TrainLogInfo:
    samples: int
    scalar_logs_dict: Any
    dist_logs_dict: Any
    image_logs_dict: Any = None


@dataclass
class LogParams:
    log_dir: str
    agent_steps_per_log: int = 10_000
    train_samples_per_log: int = 100_000
    log_on: LogOn = LogOn.TRAIN_SAMPLES


def extract_images_to_log(obs: torch.tensor, N: int = 3):
    assert len(obs.shape) == 4
    if obs.shape[1] == 3:
        I = obs[0:N, 0:3, :, :]  # This will appear as RGB
    else:
        I = obs[0:N, :, :, :].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    return I


class Logger:
    def __init__(
        self,
        experiment_path: str,
        agent_steps_per_log: str,
        train_samples_per_log: str,
        log_on: LogOn = LogOn.TRAIN_SAMPLES,
    ):
        self._log_on = log_on
        self._agent_steps_per_log = agent_steps_per_log
        self._train_samples_per_log = train_samples_per_log
        self._episode_infos_buffer = []

        self._agent_sps = 0
        self._train_sps = 0
        self._num_episodes = 0
        self._agent_steps = 0
        self._update_steps = 0
        self._train_samples = 0
        self._last_log_agent_steps = 0
        self._last_log_train_samples = 0

        self._time_delta = TimeDelta()
        self._writer = SummaryWriter(experiment_path)

    def log_hyper_params(self, hp):
        pass

    def log_model(self, model, inputs=None):
        self._writer.add_graph(model, input_to_model=inputs)

    def update(
        self,
        agent_steps,
        update_steps,
        train_samples,
        episode_infos,
        train_info,
    ):
        # check this prior to updating agent_steps/train_samples.
        ready_for_log = self.should_log()

        self._agent_steps = agent_steps
        self._update_steps = update_steps
        self._train_samples = train_samples
        self._episode_infos_buffer.extend(episode_infos)

        if ready_for_log:
            delta_time = self._time_delta.get_delta_in_seconds()
            self._agent_sps = float(agent_steps - self._last_log_agent_steps) / delta_time
            self._train_sps = float(train_samples - self._last_log_train_samples) / delta_time
            self._last_log_agent_steps = agent_steps
            self._last_log_train_samples = train_samples
            self._time_delta.update()

            # TODO move train_info to train_infos buffer
            self._log(train_info)
            self._writer.flush()

    def _log(self, train_info):
        agent_steps = self._agent_steps

        train_samples = self._train_samples
        time_string = self._time_delta.get_total_time_as_string()

        num_infos = len(self._episode_infos_buffer)
        if num_infos > 0:
            self._num_episodes += num_infos
            rewards = [x.reward for x in self._episode_infos_buffer]
            num_steps = [x.steps for x in self._episode_infos_buffer]
            mean_reward = sum(rewards) / float(num_infos)
            mean_episode_length = float(sum(num_steps)) / float(num_infos)

            # self._writer.add_histogram('rewards/reward', np.asarray(self._episode_infos_buffer[0].rewards), agent_steps)
            self._episode_infos_buffer = []

            self._writer.add_scalar("rewards/reward", mean_reward, agent_steps)
            self._writer.add_scalar("rewards/reward_vs_ep", mean_reward, self._num_episodes)
            self._writer.add_scalar("environment/episode_length", mean_episode_length, agent_steps)
            reward_string = f"R: {mean_reward:.1f}"
            ep_len_string = f"EL: {mean_episode_length:.1f}"
        else:
            reward_string = "R: --.--"
            ep_len_string = "EL: --"

        print(
            time_string
            + ", "
            + reward_string
            + ", "
            + ep_len_string
            + f", SAMP: {self._train_samples}, STEPS: {self._agent_steps}"
            + f", SPS: {self._agent_sps:.1f}, BPS: {self._train_sps:.1f}"
        )
        self._writer.add_scalar("training_stats/train_samples", train_samples, agent_steps)

        self._writer.add_scalar("performance/agent_steps_per_second", self._agent_sps, agent_steps)
        self._writer.add_scalar("performance/samples_per_second", self._train_sps, agent_steps)

        for k, v in train_info.scalar_logs_dict.items():
            self._writer.add_scalar(k, v, agent_steps)

        for k, v in train_info.dist_logs_dict.items():
            self._writer.add_histogram(k, v, agent_steps)

        for k, v in train_info.image_logs_dict.items():
            self._writer.add_images(k, v, agent_steps)

    def should_log(self) -> bool:
        if self._log_on == LogOn.TRAIN_SAMPLES:
            return (self._train_samples - self._last_log_train_samples) > self._train_samples_per_log
        else:
            return (self._agent_steps - self._last_log_agent_steps) > self._agent_steps_per_log

    def stop(self):
        self._writer.close()

    def start_timers(self):
        self._time_delta.reset()


def log_grads_and_vars(log_name, named_parameters, scalar_logs, dist_log_dict):
    for n, v in named_parameters:
        if (v.requires_grad) and ("bias" not in n):
            scalar_logs[log_name + "_grads/" + n] = v.grad.abs().sum()
            scalar_logs[log_name + "_params/" + n] = v.abs().mean()
            dist_log_dict[log_name + "_params/" + n] = v[:].detach().cpu().numpy()
            # dist_log_dict["policy_grads/"+n] = v.grad[:].detach().cpu().numpy()
