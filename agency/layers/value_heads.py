import torch
import torch.nn as nn
from agency.layers.feed_forward import custom_ortho_init_


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


class QVisionEncoder(nn.Module):
    def __init__(self, conv_encoder, mlp_encoder):
        super().__init__()
        self._conv_encoder = conv_encoder
        self._mlp_encoder = mlp_encoder

    def forward(self, state, actions=None):
        x = self._conv_encoder(state)
        if actions is not None:
            x = torch.cat((x, actions), dim=1)
        return self._mlp_encoder(x)


class QEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self._encoder = encoder

    def forward(self, state, actions=None):
        if actions is not None:
            state = torch.cat((state, actions), dim=1)
        return self._encoder(state)


class QHead(nn.Module):
    def __init__(self, encoder, input_size, output_size=1):
        super().__init__()
        self._encoder = encoder
        self._final_layer = nn.Linear(input_size, output_size)

        custom_ortho_init_(self._final_layer.weight, gain=1.0)
        nn.init.constant_(self._final_layer.bias, 0.0)

    def forward(self, state, actions=None):
        if actions is not None:
            return self._final_layer(self._encoder(state, actions))
        else:
            return self._final_layer(self._encoder(state))


class VHead(nn.Module):
    def __init__(self, encoder, input_size):
        super().__init__()
        self._encoder = encoder
        self._final_layer = nn.Linear(input_size, 1)

        custom_ortho_init_(self._final_layer.weight, gain=1.0)
        nn.init.constant_(self._final_layer.bias, 0.0)

    def forward(self, state):
        return self._final_layer(self._encoder(state))
