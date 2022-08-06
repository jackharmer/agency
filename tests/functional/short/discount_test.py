import torch
from agency.tools.gamma_matrix import discount, make_gamma_matrix
from pytest import approx


def simple_discount(rewards, gamma, value, masks):
    discounts = []
    V = value
    for cc in reversed(range(len(rewards))):
        V = rewards[cc] + gamma * masks[cc] * V
        discounts.append(V)
    return list(reversed(discounts))


def get_masks():
    return [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]


def test_simple_discount_works():
    rewards = [0.1, 0.2, 0.3, 0.4]
    masks = get_masks()
    gamma = 0.99
    value = 10

    for mask in masks:
        # Create the true values
        v3 = rewards[3] + gamma * mask[3] * value
        v2 = rewards[2] + gamma * mask[2] * v3
        v1 = rewards[1] + gamma * mask[1] * v2
        v0 = rewards[0] + gamma * mask[0] * v1

        d_true = [v0, v1, v2, v3]

        d = simple_discount(rewards, gamma, value, mask)

        assert d_true == approx(d, 1e-5)


def test_gamma_matrix_discount():
    rewards = [0.1, 0.2, 0.3, 0.4]
    masks = get_masks()
    gamma = 0.99
    value = 10
    gamma_matrix = make_gamma_matrix(gamma, len(rewards))

    for mask in masks:
        d_simple = simple_discount(rewards, gamma, value, mask)
        d_gamma = discount(
            torch.tensor(rewards).unsqueeze(0),
            torch.tensor([value * mask[-1]]).unsqueeze(0),
            gamma_matrix,
            torch.tensor(mask).unsqueeze(0),
        )
        assert d_gamma.cpu().numpy() == approx(d_simple, 1e-5)
