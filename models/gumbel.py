import torch
import torch.nn.functional as F
import numpy as np


def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * torch.log(-torch.log(y))

def gumbel_softmax_sampling(h, dim=-1, mu=0, beta=1, tau=0.1):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    device = h.device  # Get the device of the input tensor h
    shape_h = h.shape
    p = F.softmax(h, dim=dim)
    y = torch.rand(shape_h, device=device) + 1e-25  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta)
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    x = x / tau
    x = F.softmax(x, dim=dim)  # now, the x approximates a one_hot vector.
    return x
