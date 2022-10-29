from dataclasses import dataclass
import dataclasses
from enum import Enum
from typing import Any

import torch
from agency.tools.timer import TimeDelta
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except:
    pass


class LogOn(Enum):
    TRAIN_SAMPLES = 1
    AGENT_STEPS = 2


@dataclass
class TrainLogInfo:
    samples: int
    scalar_logs_dict: Any
    dist_logs_dict: Any
    image_logs_dict: Any = None
    video_logs_dict: Any = None


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
        log_to_terminal: bool = True,
        use_wandb: bool = False,
        wandb_project_name: str = "unknown",
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
        self._log_to_terminal = log_to_terminal

        self._time_delta = TimeDelta()
        self._use_wandb = use_wandb

        if self._use_wandb:
            wandb.init(project=wandb_project_name, config={"name": wandb_project_name}, save_code=True)
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        else:
            self._writer = SummaryWriter(experiment_path)

    def log_hyper_params(self, hp):
        if self._use_wandb:
            hp_dict = dataclasses.asdict(hp)
            wandb.config.update(hp_dict)

    def log_model(self, model, inputs=None):
        if self._use_wandb:
            wandb.watch(model, log_freq=100, log_graph=False, log="all")
        else:
            self._writer.add_graph(model, input_to_model=inputs)

    def update(
        self,
        agent_steps,
        update_steps,
        train_samples,
        episode_infos,
        train_info: TrainLogInfo,
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

            if not self._use_wandb:
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
            videos = [x.video_data for x in self._episode_infos_buffer if x.video_data is not None]
            mean_reward = sum(rewards) / float(num_infos)
            mean_episode_length = float(sum(num_steps)) / float(num_infos)

            # self._writer.add_histogram('rewards/reward', np.asarray(self._episode_infos_buffer[0].rewards), agent_steps)
            self._episode_infos_buffer = []

            if len(videos) > 0:
                video: torch.Tensor = torch.stack(videos, dim=0)

            if self._use_wandb:
                wandb.log(
                    {
                        "rewards/reward": mean_reward,
                        "rewards/reward_vs_ep": mean_reward,
                        "environment/episode_length": mean_episode_length,
                    },
                    step=agent_steps,
                )
                if len(videos) > 0:
                    wandb.log({"videos/episode": wandb.Video(video.cpu().numpy(), fps=30)}, step=agent_steps)
            else:
                self._writer.add_scalar("rewards/reward", mean_reward, agent_steps)
                self._writer.add_scalar("rewards/reward_vs_ep", mean_reward, self._num_episodes)
                self._writer.add_scalar("environment/episode_length", mean_episode_length, agent_steps)
                if len(videos) > 0:
                    self._writer.add_video("videos/episode", video, agent_steps, fps=30)
            reward_string = f"R: {mean_reward:.1f}"
            ep_len_string = f"EL: {mean_episode_length:.1f}"
        else:
            reward_string = "R: --.--"
            ep_len_string = "EL: --"

        if self._log_to_terminal:
            print(
                time_string
                + ", "
                + reward_string
                + ", "
                + ep_len_string
                + f", SAMP: {self._train_samples}, STEPS: {self._agent_steps}"
                + f", SPS: {self._agent_sps:.1f}, BPS: {self._train_sps:.1f}"
            )
        if self._use_wandb:
            wandb.log(
                {
                    "training_stats/agent_steps": self._agent_steps,
                    "training_stats/train_samples": train_samples,
                    "performance/agent_steps_per_second": self._agent_sps,
                    "performance/samples_per_second": self._train_sps,
                },
                step=agent_steps,
            )
            wandb.log(train_info.scalar_logs_dict, step=agent_steps)

            # for k, v in train_info.dist_logs_dict.items():
            #     self._writer.add_histogram(k, v, agent_steps)

            if train_info.image_logs_dict is not None:
                wandb.log(
                    {k: wandb.Image(v.float()) for k, v in train_info.image_logs_dict.items()},
                    step=agent_steps,
                )

            if train_info.video_logs_dict is not None:
                wandb.log(
                    {k: wandb.Video(v.cpu().numpy(), fps=30) for k, v in train_info.video_logs_dict.items()},
                    step=agent_steps,
                )

        else:
            self._writer.add_scalar("training_stats/train_samples", train_samples, agent_steps)
            self._writer.add_scalar("performance/agent_steps_per_second", self._agent_sps, agent_steps)
            self._writer.add_scalar("performance/samples_per_second", self._train_sps, agent_steps)

            for k, v in train_info.scalar_logs_dict.items():
                self._writer.add_scalar(k, v, agent_steps)

            # for k, v in train_info.scalar_logs_dict.items():
            #     self._writer.add_scalar(k + "_vs_samples", v, train_samples)

            for k, v in train_info.dist_logs_dict.items():
                self._writer.add_histogram(k, v, agent_steps)

            if train_info.image_logs_dict is not None:
                for k, v in train_info.image_logs_dict.items():
                    self._writer.add_images(k, v, agent_steps)

            if train_info.video_logs_dict is not None:
                for k, v in train_info.video_logs_dict.items():
                    self._writer.add_video(k, v, agent_steps, fps=30)

            # for k, v in train_info.image_logs_dict.items():
            #     self._writer.add_images(k + "_vs_samples", v, train_samples)

    def should_log(self) -> bool:
        if self._log_on == LogOn.TRAIN_SAMPLES:
            return (self._train_samples - self._last_log_train_samples) > self._train_samples_per_log
        else:
            return (self._agent_steps - self._last_log_agent_steps) > self._agent_steps_per_log

    def stop(self):
        if not self._use_wandb:
            self._writer.close()

    def start_timers(self):
        self._time_delta.reset()


def log_grads_and_vars(
    log_name,
    named_parameters,
    scalar_logs,
    dist_log_dict=None,
    log_dists: bool = False,
    lr: float = None,
):
    g_name = "grads/" + log_name + "/"
    gw_name = "grads_weight_ratio/" + log_name + "/"
    gws_name = "grads_weight_ratio_scaled/" + log_name + "/"
    w_name = "weights/" + log_name + "/"

    for n, w in named_parameters:
        is_bias = "bias" in n
        if (w.requires_grad) and not is_bias:
            w_abs = w.abs()
            scalar_logs[w_name + n] = w_abs.mean()

            if w.grad is None:
                scalar_logs[g_name + n] = 0
            else:
                g_abs = w.grad.abs()
                scalar_logs[g_name + n] = g_abs.mean()

            if w.grad is not None:
                gw_ratio: torch.Tensor = (g_abs / w_abs).mean()
                scalar_logs[gw_name + n] = gw_ratio
                if lr is not None:
                    scalar_logs[gws_name + n] = (lr * gw_ratio).log10()

            if dist_log_dict is not None and log_dists:
                dist_log_dict[w_name + n] = w[:].detach().cpu().numpy()
                # dist_log_dict["policy_grads/"+n] = v.grad[:].detach().cpu().numpy()
