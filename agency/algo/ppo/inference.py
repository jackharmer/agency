from dataclasses import dataclass
from typing import Any

import torch
from agency.memory.block_memory import AgentStep, BlockOfSteps


@dataclass
class PpoAuxData:
    value: torch.Tensor

    def __getitem__(self, key):
        return PpoAuxData(value=self.value[key])


@dataclass
class PpoInferer:
    net: Any
    num_actions: int
    device: Any
    return_aux_data: bool = True

    def infer(self, obs, random_actions: bool = False) -> Any:
        obs = obs.to(self.device)

        if random_actions:
            with torch.no_grad():
                policy = self.net.policy.random(obs.shape[0])
        else:
            with torch.no_grad():
                policy = self.net.policy(obs)

        if self.return_aux_data:
            with torch.no_grad():
                aux_data = PpoAuxData(value=self.net.value(obs).detach())
            return policy.detach(), aux_data
        else:
            return policy.detach()
