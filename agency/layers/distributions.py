from dataclasses import dataclass
import math
from re import I
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from agency.layers.feed_forward import custom_ortho_init_


@dataclass
class ContinuousDistParams:
    categorical: bool = False


@dataclass
class DiscreteDistParams:
    categorical: bool = False
    temperature: float = 4
    hard: bool = False


class BasePolicyOutput:
    def _get_members(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def _apply_to_each_member_variable(self, func):
        for member in self._get_members():
            self.__dict__[member] = self._apply_to_variable_if_not_none(func, self.__dict__[member])
        return self

    def _apply_to_variable_if_not_none(self, func, val):
        if val is None:
            return None
        return func(val)

    def _create_new(self, func=lambda x: x):
        raise NotImplementedError()

    def detach(self):
        return self._apply_to_each_member_variable(lambda x: x.detach())

    def cpu(self):
        return self._apply_to_each_member_variable(lambda x: x.cpu())

    def numpy(self):
        return self._apply_to_each_member_variable(lambda x: x.numpy())

    def __getitem__(self, key):
        return self._create_new(lambda x: x[key].unsqueeze(0))
        # return self._create_new(lambda x: np.expand_dims(x[key], axis=0)) # numpy


class CategoricalPolicySample(BasePolicyOutput):
    def __init__(
        self,
        sample: Any,
        probs: Any = None,
        log_prob: Any = None,
        entropy: Any = None,
    ):
        self.sample = sample
        self.probs = probs
        self.log_prob = log_prob
        self.entropy = entropy

    def _create_new(self, func=lambda x: x):
        return CategoricalPolicySample(
            sample=func(self.sample),
            probs=self._apply_to_variable_if_not_none(func, self.probs),
            log_prob=self._apply_to_variable_if_not_none(func, self.log_prob),
            entropy=self._apply_to_variable_if_not_none(func, self.entropy),
        )


class DiscreteGumbelPolicySample(BasePolicyOutput):
    def __init__(
        self,
        sample: Any,
        log_prob: Any = None,
        logits: Any = None,
        entropy: Any = None,
    ):
        self.sample = sample
        self.log_prob = log_prob
        self.logits = logits
        self.entropy = entropy

    def _create_new(self, func=lambda x: x):
        return DiscreteGumbelPolicySample(
            sample=func(self.sample),
            log_prob=self._apply_to_variable_if_not_none(func, self.log_prob),
            logits=self._apply_to_variable_if_not_none(func, self.logits),
            entropy=self._apply_to_variable_if_not_none(func, self.entropy),
        )


class GaussianPolicySample(BasePolicyOutput):
    def __init__(
        self,
        sample: Any = None,
        mu: Any = None,
        std: Any = None,
        log_prob: Any = None,
        entropy: Any = None,
    ):
        self.sample = sample
        self.mu = mu
        self.std = std
        self.log_prob = log_prob
        self.entropy = entropy

    def _create_new(self, func=lambda x: x):
        return GaussianPolicySample(
            sample=self._apply_to_variable_if_not_none(func, self.sample),
            mu=self._apply_to_variable_if_not_none(func, self.mu),
            std=self._apply_to_variable_if_not_none(func, self.std),
            log_prob=self._apply_to_variable_if_not_none(func, self.log_prob),
            entropy=self._apply_to_variable_if_not_none(func, self.entropy),
        )


class ClippedTanhTransform(TanhTransform):
    EPS = 1e-6

    def _inverse(self, y):
        return torch.atanh(y.clamp(-1 + self.EPS, 1 - self.EPS))


class TanhDiagNormal(TransformedDistribution):
    EPS = 1e-6

    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        self._base_dist = Independent(Normal(mu, std), 1)
        super().__init__(self._base_dist, [ClippedTanhTransform()])

    def rsample(self, log_prob=False):
        orig_sample = super().rsample()
        sample = orig_sample.clamp(-1 + self.EPS, 1 - self.EPS)
        if log_prob:
            return sample, self.log_prob(sample)
        else:
            return sample

    @property
    def mean(self):
        return torch.tanh(self._base_dist.mean).clamp(-1 + self.EPS, 1 - self.EPS)

    def entropy(self):
        log_prob = self.log_prob(self.rsample())
        return -log_prob.mean(0)


class PredStdLayer(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        out_gain: float = 0.01,
        state_independent_std: bool = False,
        state_independent_init: float = 0.5,
    ):
        super().__init__()
        self._state_independent_std = state_independent_std
        if state_independent_std:
            self._pre_std = torch.nn.Parameter(
                state_independent_init * torch.ones(num_outputs), requires_grad=True
            )
        else:
            self._pre_std = nn.Linear(num_inputs, num_outputs)
            nn.init.xavier_uniform_(self._pre_std.weight, gain=out_gain)
            nn.init.constant_(self._pre_std.bias, 0.0)

    def forward(self, state: torch.Tensor):
        if self._state_independent_std:
            return self._pre_std.repeat(state.shape[0], 1)
        else:
            return self._pre_std(state)


class StdLayerExp(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        state_independent_std: bool = False,
        min_std: float = 0.001,
        max_std: float = 2.0,
        out_gain: float = 0.01,
    ):
        super().__init__()
        self._log_min_std = math.log(min_std)
        self._log_max_std = math.log(max_std)
        self._pre_std = PredStdLayer(num_inputs, num_outputs, out_gain, state_independent_std, -0.7)

    def forward(self, state: torch.Tensor):
        pre_std = self._pre_std(state)
        std = torch.exp(pre_std.clamp(self._log_min_std, self._log_max_std))
        return std


class StdLayerSoftPlus(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        state_independent_std: bool = False,
        min_std: float = 0.001,
        max_std: float = 2.0,
        out_gain: float = 0.01,
    ):
        super().__init__()
        self._min_std = min_std
        self._max_std = max_std
        self._pre_std = PredStdLayer(num_inputs, num_outputs, out_gain, state_independent_std)

    def forward(self, state: torch.Tensor):
        pre_std = self._pre_std(state)
        std = F.softplus(pre_std.clamp)
        std = (std + self._min_std).clamp(max=self._max_std)
        return std


class StdLayerSigmoid(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        state_independent_std: bool = False,
        min_std: float = 0.001,
        max_std: float = 2.0,
        out_gain: float = 0.01,
    ):
        super().__init__()
        self._min_std = min_std
        self._max_std = max_std
        self._pre_std = PredStdLayer(num_inputs, num_outputs, out_gain, state_independent_std)

    def forward(self, state: torch.Tensor):
        pre_std = self._pre_std(state)
        std = self._max_std * F.sigmoid(pre_std / self._max_std) + self._min_std
        return std


class GaussianPolicy(nn.Module):
    DIST_CLASS = TanhDiagNormal

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        state_independent_std: bool = False,
        min_std=0.0001,
        max_std=5.0,
        mu_limit=None,
        out_gain=0.01,
        std_class=StdLayerExp,
    ):
        super().__init__()
        self._num_actions = num_actions
        self._mu_limit = mu_limit
        self._eps = 1e-8
        self._action_eps = 1e-3

        self._mu = nn.Linear(num_inputs, num_actions)
        self._std = std_class(
            num_inputs,
            num_actions,
            state_independent_std=state_independent_std,
            min_std=min_std,
            max_std=max_std,
            out_gain=out_gain,
        )

        nn.init.xavier_uniform_(self._mu.weight, gain=out_gain)
        nn.init.constant_(self._mu.bias, 0.0)

    def forward(self, state: torch.Tensor) -> GaussianPolicySample:
        mu = self._mu(state)
        if self._mu_limit is not None:
            mu = self._mu_limit * F.tanh(mu / self._mu_limit)

        std = self._std(state)

        dist = self.DIST_CLASS(mu, std)
        sample, log_prob = dist.rsample(log_prob=True)

        sample_entropy = -log_prob

        policy_sample = GaussianPolicySample(
            sample=sample,
            mu=mu,
            std=std,
            log_prob=log_prob,
            entropy=sample_entropy,
        )
        return policy_sample

    def random(self, batch_size: int):
        sample = torch.rand(batch_size, self._num_actions) * 2.0 - 1.0
        sample = torch.clamp(sample, -1 + self._action_eps, 1 - self._action_eps)
        return GaussianPolicySample(sample=sample)

    def log_prob_of_sample(self, sample: torch.Tensor, policy: GaussianPolicySample):
        return self.make_distribution(policy).log_prob(sample)

    def make_distribution(self, policy: GaussianPolicySample):
        return self.DIST_CLASS(policy.mu, policy.std)

    def make_batch(self, list_of_policy_samples: list[GaussianPolicySample]):
        return GaussianPolicySample(
            sample=torch.cat([other.sample for other in list_of_policy_samples]),
            mu=torch.cat([other.mu for other in list_of_policy_samples]),
            std=torch.cat([other.std for other in list_of_policy_samples]),
            log_prob=torch.cat([other.log_prob for other in list_of_policy_samples]),
            entropy=torch.cat([other.entropy for other in list_of_policy_samples]),
        )


class DiscretePolicy(nn.Module):
    """
    See:
    https://arxiv.org/pdf/1611.01144.pdf

    https://medium.com/mini-distill/discrete-optimization-beyond-reinforce-5ca171bebf17

    params:
        num_inputs: (int)
        num_actions: (int)
        temperature: gumbel temperature parameter.
    """

    def __init__(self, num_inputs, num_actions, temperature, hard=False):
        super().__init__()
        self._num_actions = num_actions
        self._temperature = temperature
        self._hard = hard

        self._linear = nn.Linear(num_inputs, num_actions)

        custom_ortho_init_(self._linear.weight, gain=0.01)
        nn.init.constant_(self._linear.bias, 0.0)

    def forward(self, state) -> DiscreteGumbelPolicySample:
        logits = self._linear(state)

        dist = torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.tensor(self._temperature), logits=logits
        )
        y_soft = dist.rsample()
        log_prob = dist.log_prob(y_soft)

        if self._hard:
            # Straight through.
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
                -1, index, 1.0
            )
            sample = y_hard - y_soft.detach() + y_soft
        else:
            sample = y_soft

        # entropy = dist.entropy() # Not implemented for RelaxedOneHotCategorical
        entropy = -log_prob

        return DiscreteGumbelPolicySample(sample=sample, log_prob=log_prob, logits=logits, entropy=entropy)

    def random(self, batch_size):
        sample = torch.randint(self._num_actions, (batch_size, 1))
        return DiscreteGumbelPolicySample(sample=sample)

    def log_prob_of_sample(self, sample, policy):
        return self.make_distribution(policy).log_prob(sample)

    def make_distribution(self, policy):
        return torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.tensor(self._temperature),
            logits=policy.logits,
        )

    def make_batch(self, list_of_policy_samples):
        return DiscreteGumbelPolicySample(
            sample=torch.cat([other.sample for other in list_of_policy_samples]),
            log_prob=torch.cat([other.log_prob for other in list_of_policy_samples]),
            logits=torch.cat([other.logits for other in list_of_policy_samples]),
            entropy=torch.cat([other.entropy for other in list_of_policy_samples]),
        )


class CategoricalPolicy(nn.Module):
    """
    params:
        num_inputs: (int)
        num_actions: (int)
    """

    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self._num_actions = num_actions
        self._linear = nn.Linear(num_inputs, num_actions)

        custom_ortho_init_(self._linear.weight, gain=0.01)
        nn.init.constant_(self._linear.bias, 0.0)

    def forward(self, state) -> CategoricalPolicySample:
        logits = self._linear(state)
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample_one_hot = dist.sample()

        policy = CategoricalPolicySample(
            sample=sample_one_hot,
            probs=dist.probs,
            log_prob=dist.log_prob(sample_one_hot),
            entropy=dist.entropy(),
        )
        return policy

    def random(self, batch_size):
        sample = torch.randint(self._num_actions, (batch_size, 1))
        sample_one_hot = F.one_hot(sample, num_classes=self._num_actions)
        return CategoricalPolicySample(sample_one_hot=sample_one_hot, sample=sample)

    def log_prob_of_sample(self, sample, policy):
        return self.make_distribution(policy).log_prob(sample)

    def make_distribution(self, policy):
        return torch.distributions.OneHotCategorical(probs=policy.probs)

    def make_batch(self, list_of_policy_samples):
        return CategoricalPolicySample(
            sample=torch.cat([other.sample for other in list_of_policy_samples]),
            probs=torch.cat([other.probs for other in list_of_policy_samples]),
            log_prob=torch.cat([other.log_prob for other in list_of_policy_samples]),
            entropy=torch.cat([other.entropy for other in list_of_policy_samples]),
        )
