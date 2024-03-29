import torch.nn as nn


class PolicyHead(nn.Module):
    def __init__(self, encoder, policy):
        super().__init__()
        self._encoder = encoder
        self._policy = policy

    def forward(self, state):
        enc = self._encoder(state)
        policy = self._policy(enc)
        return policy

    def log_prob_of_sample(self, sample, policy):
        return self._policy.log_prob_of_sample(sample, policy)

    def make_batch(self, list_of_policies):
        return self._policy.make_batch(list_of_policies)

    def make_distribution(self, policy):
        return self._policy.make_distribution(policy)

    def random(self, batch_size):
        return self._policy.random(batch_size)
