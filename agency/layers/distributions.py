import math
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
from agency.layers.feed_forward import custom_ortho_init_


class BasePolicyOutput:
    def _apply_func_to_members(self, func):
        for member in self._get_members():
            self.__dict__[member] = self._apply_func_if_not_none(func, self.__dict__[member])
        return self

    def _create_new(self, func=lambda x: x):
        raise NotImplementedError()

    def _get_members(self):
        return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    def detach(self):
        return self._apply_func_to_members(lambda x: x.detach())

    def cpu(self):
        return self._apply_func_to_members(lambda x: x.cpu())

    def numpy(self):
        return self._apply_func_to_members(lambda x: x.numpy())

    def torch(self):
        return self._apply_func_to_members(lambda x: torch.tensor(x, dtype=torch.float32))

    def copy(self):
        # This slows things down by 40%, is it needed?
        return self._create_new(np.copy)
        # return self._create_new(lambda x: x.copy())

    def split_batch(self):
        return [self._create_new(lambda x: x[cc].unsqueeze(0)) for cc in range(self.sample.shape[0])]

    def split_batch_np(self):
        return [self._create_new(lambda x: np.expand_dims(x[cc], axis=0)) for cc in range(self.sample.shape[0])]

    def clone(self):
        return self._create_new(lambda x: x.clone())

    def _apply_func_if_not_none(self, func, val):
        if val is None:
            return None
        return func(val)


class CategoricalPolicySample(BasePolicyOutput):
    def __init__(
            self,
            sample: Any,
            prob: Any = None,
            entropy: Any = None,
    ):
        self.sample = sample
        self.prob = prob
        self.entropy = entropy

    def _create_new(self, func=lambda x: x):
        return CategoricalPolicySample(
            sample=func(self.sample),
            prob=self._apply_func_if_not_none(func, self.prob),
            entropy=self._apply_func_if_not_none(func, self.entropy),
        )


class GaussianPolicySample(BasePolicyOutput):
    def __init__(
            self,
            sample: Any,
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
            sample=func(self.sample),
            mu=self._apply_func_if_not_none(func, self.mu),
            std=self._apply_func_if_not_none(func, self.std),
            log_prob=self._apply_func_if_not_none(func, self.log_prob),
            entropy=self._apply_func_if_not_none(func, self.entropy),
        )


class TanhDiagNormal(TransformedDistribution):
    def __init__(self, mu, std):
        self._base_dist = Independent(Normal(mu, std), 1)
        super().__init__(
            self._base_dist,
            [TanhTransform()]
        )

    def rsample(self, log_prob=False):
        sample = super().rsample()
        if log_prob:
            return sample, self.log_prob(sample)
        else:
            return sample

    def entropy(self):
        # TODO
        return self._base_dist.entropy()


class TanhDiagNormalCustom:
    EPS = 1e-8

    def __init__(self, mu, std):
        self._dist = Normal(mu, std)

    def rsample(self, log_prob=False):
        raw_sample = self._dist.rsample()
        sample = torch.tanh(raw_sample)
        if log_prob:
            return sample, self.log_prob(sample, raw_sample)
        else:
            return sample

    def log_prob(self, sample, raw_sample):
        raw_log_prob = self._dist.log_prob(raw_sample).sum(axis=-1)
        squash_correction = torch.log(1.0 - sample.pow(2.0) + self.EPS).sum(axis=1)
        corrected_log_prob = raw_log_prob - squash_correction
        return corrected_log_prob

    def entropy(self):
        return self._dist.entropy()


class DiscreteGumbelPolicySample(BasePolicyOutput):
    def __init__(
            self,
            sample: Any,
            log_prob: Any = None,
            logits: Any = None,
    ):
        self.sample = sample
        self.log_prob = log_prob
        self.logits = logits

    def _modify_in_place(self, func):
        self.sample = func(self.sample)
        self.log_prob = self._apply_func_if_not_none(func, self.log_prob)
        self.logits = self._apply_func_if_not_none(func, self.logits)
        return self

    def _create_new(self, func=lambda x: x):
        return DiscreteGumbelPolicySample(
            sample=func(self.sample),
            log_prob=self._apply_func_if_not_none(func, self.log_prob),
            logits=self._apply_func_if_not_none(func, self.logits),
        )


class GaussianPolicy(nn.Module):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    # DIST_CLASS = TanhDiagNormal
    DIST_CLASS = TanhDiagNormalCustom
    EPS = 1e-8
    ACTION_EPS = 1e-3

    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self._num_actions = num_actions
        self._mu = nn.Linear(num_inputs, num_actions)
        self._log_std = nn.Linear(num_inputs, num_actions)

        nn.init.xavier_uniform_(self._mu.weight, gain=0.01)
        nn.init.xavier_uniform_(self._log_std.weight, gain=0.01)
        nn.init.constant_(self._mu.bias, 0.0)
        nn.init.constant_(self._log_std.bias, 0.0)

    def log_prob_of_sample(self, sample, policy):
        dist = self.DIST_CLASS(policy.mu, policy.std)
        raw_sample = torch.atanh(torch.clamp(sample, -1 + self.ACTION_EPS, 1 - self.ACTION_EPS))
        return dist.log_prob(sample, raw_sample)

    def forward(self, state) -> GaussianPolicySample:
        mu = self._mu(state)
        log_std = torch.clamp(self._log_std(state), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = self.DIST_CLASS(mu, std)
        sample, log_prob = dist.rsample(log_prob=True)

        entropy = -log_prob

        policy_sample = GaussianPolicySample(
            sample=sample,
            mu=mu,
            std=std,
            log_prob=log_prob,
            entropy=entropy
        )
        return policy_sample

    def random(self, batch_size):
        sample = torch.rand(batch_size, self._num_actions) * 2.0 - 1.0
        sample = torch.clamp(sample, -1 + self.ACTION_EPS, 1 - self.ACTION_EPS)
        return GaussianPolicySample(sample=sample)


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
            temperature=torch.tensor(self._temperature),
            logits=logits
        )
        y_soft = dist.rsample()
        log_prob = dist.log_prob(y_soft)

        if self._hard:
            # Straight through.
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            sample = y_hard - y_soft.detach() + y_soft
        else:
            sample = y_soft

        return DiscreteGumbelPolicySample(sample=sample, log_prob=log_prob, logits=logits)

    def random(self, batch_size):
        sample = torch.randint(self._num_actions, (batch_size, 1))
        return DiscreteGumbelPolicySample(sample=sample)

    def log_prob_of_sample(self, sample, policy):
        dist = torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.tensor(self._temperature),
            logits=policy.logits
        )
        return dist.log_prob(sample)


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
            prob=dist.probs,
            entropy=dist.entropy()
        )
        return policy

    def random(self, batch_size):
        sample = torch.randint(self._num_actions, (batch_size, 1))
        sample_one_hot = F.one_hot(sample, num_classes=self._num_actions)
        return CategoricalPolicySample(
            sample_one_hot=sample_one_hot,
            sample=sample
        )
