# from torch.nn.utils import clip_grad_norm_
import torch
from torch._six import inf


def calculate_conv_output_size(in_size, out_channels, kernels, strides, padding=None, dilation=None):
    if padding is None:
        padding = [0] * len(kernels)
    if dilation is None:
        dilation = [1] * len(kernels)
    out_size = list(in_size)
    for kernel, stride, pad, dil in zip(kernels, strides, padding, dilation):
        out_size[0] = ((out_size[0] + 2.0 * pad - (dil * (kernel - 1)) - 1) // stride) + 1
        out_size[1] = ((out_size[1] + 2.0 * pad - (dil * (kernel - 1)) - 1) // stride) + 1
    return int(out_size[0] * out_size[1] * out_channels)


# Modified version of pytorch clip_grad_norm, to add a clip argument, such that we get grad norms
# for logging even if we don't clip.
def clip_grad_norm_(parameters, max_norm, norm_type=2, clip=True):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        total_norm = torch.tensor(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip:
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    return total_norm


def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
