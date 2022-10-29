from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EmptyAuxData:
    def __getitem__(self, key):
        return EmptyAuxData()


@dataclass
class Inferer:
    net: Any
    num_actions: int
    device: Any
    return_aux_data: bool = True

    def infer(self, obs, random_actions: bool = False) -> Any:
        if random_actions:
            with torch.no_grad():
                policy = self.net.policy.random(obs.shape[0])
        else:
            with torch.no_grad():
                policy = self.net.policy(obs.to(self.device))

        if self.return_aux_data:
            return policy.detach(), EmptyAuxData()
        else:
            return policy.detach()
