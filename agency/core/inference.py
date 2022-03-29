from typing import Any
from dataclasses import dataclass
import torch


@dataclass
class Inferer:
    net: Any
    num_actions: int
    device: Any

    def infer(self, obs, random_actions: bool = False) -> Any:
        if random_actions:
            with torch.no_grad():
                policy = self.net.policy.random(obs.shape[0])
        else:
            with torch.no_grad():
                policy = self.net.policy(obs.to(self.device))

        return policy.detach().cpu()
