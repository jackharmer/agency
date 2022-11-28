import torch
from einops import rearrange


# Construct a gamma matrix for optimised discount calculations.
# Using this in combination with the discount() function below
# provides 100x speedup over a non gamma matrix variant.
#
# Gamma Matrix Form  [roll_length+1, roll_length]:
#    [0.99^0, 0.0,    0.0   ]
#    [0.99^1, 0.99^0, 0.0   ]
#    [0.99^2, 0.99^1, 0.99^0]
#    [0.99^3, 0.99^2, 0.99^1]
#
#
# This allow the discount to be calculated as a dot product of the
# reward matrix and the gammaMatrix in one calculation across the whole
# batch.
#
# Reward Matrix:  [num_rolls, roll_length+1]
def make_gamma_matrix(gamma, roll_length):
    gamma = torch.tensor(gamma, dtype=torch.float32)
    gamma_matrix = torch.zeros((roll_length + 1, roll_length), dtype=torch.float32)
    gamma_vector = torch.zeros((roll_length + 1), dtype=torch.float32)
    for cc in range(roll_length + 1):
        gamma_vector[cc] = pow(gamma, cc)
    for cc in range(roll_length):
        gamma_matrix[cc : (roll_length + 1), cc] = gamma_vector[0 : roll_length + 1 - cc]
    return gamma_matrix


# Calculate the discounted n step return using a gamma matrix, i.e. without for loops (see above).
# This version masks at every step and thus can deal with rollouts over episode boundaries :)
#
#  Reward Matrix         *     Gamma Matrix                 = Discount Matrix
#  [num_rolls, roll_length+1] [roll_length+1, roll_length]   [num_rolls, roll_length]
#
#  [ r0, r1, ..., v]           [0.99^0, 0.0   ]
#  [ r0, r1, ..., v]     *     [0.99^1, 0.99^0]
#  [ r0, r1, ..., v]           [0.99^2, 0.99^1]
# @torch.jit.script
def discount(
    rewards: torch.Tensor,  # [num_rolls, roll_length]
    values: torch.Tensor,  # [num_rolls, 1]
    gamma_matrix: torch.Tensor,  # [roll_length + 1, roll_length]
    masks: torch.Tensor,  # [num_rolls, roll_length]
):
    num_rolls = rewards.shape[0]
    roll_length = rewards.shape[1]

    # [num_rolls, roll_length + 1]
    reward_matrix = torch.cat([rewards, values], dim=1)

    # [num_rolls, roll_length + 1]
    mask_matrix = torch.cat([torch.ones((num_rolls, 1), device=rewards.device), masks], dim=1)
    # mask the gamma matrix by the terminal states, flooding the state masking to the end of each rollout
    # when calculating for each state.
    mask_matrix_rep = mask_matrix.unsqueeze(2).tile(1, 1, roll_length)
    mask_matrix_tril = 1.0 - torch.cumsum(torch.tril(1 - mask_matrix_rep, -1), 1).clamp(max=1.0)
    masked_gamma_matrix = torch.mul(mask_matrix_tril, gamma_matrix)

    # [num_rolls, roll_length]
    discount_matrix = torch.bmm(
        torch.transpose(masked_gamma_matrix, 1, 2),
        reward_matrix.unsqueeze(2),
    ).view(num_rolls, roll_length)

    # Discount vector: [num_rolls * roll_length]
    out = torch.reshape(discount_matrix, (discount_matrix.shape[0] * discount_matrix.shape[1], 1))

    return out


def gae_discount(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    terminal_mask: torch.Tensor,
    gamma: float = 0.99,
    lamb: float = 0.97,
):
    roll_length = rewards.shape[1]
    values = rearrange(values, "(b r) 1 -> b r", r=roll_length)
    next_values = torch.cat([values[:, 1:], last_value], 1)
    advantage = (rewards + gamma * terminal_mask * next_values) - values
    gae_list = []
    curr_g_adv = advantage[:, -1]
    for i in reversed(range(roll_length)):
        curr_g_adv = advantage[:, i] + lamb * gamma * terminal_mask[:, i] * curr_g_adv
        gae_list.append(curr_g_adv)
    gae_advantage = torch.stack(list(reversed(gae_list)), dim=1)
    v_bootstrap = gae_advantage + values
    return v_bootstrap.view(-1, 1)
