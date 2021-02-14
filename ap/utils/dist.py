import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def get_diag_cov(var):

    if len(var.size()) == 2:
        return torch.stack([torch.diag(x) for x in var.unbind(0)], dim=0)
    elif len(var.size()) == 3:
        return torch.stack([torch.stack([torch.diag(y) for y in x.unbind(0)]) for x in var.unbind(0)], dim=0)
    else:
        raise ValueError("Either diag var of rank 2 or 3.")


def get_normal_dist(mean, diag_var):

    cov = get_diag_cov(diag_var)
    return MultivariateNormal(mean, cov)


def sample_normal(dist, no_sample=False):

    if no_sample:
        return dist.mean
    else:
        return dist.rsample()


def softplus_inverse(x):

    return np.log(np.exp(x) - 1)
