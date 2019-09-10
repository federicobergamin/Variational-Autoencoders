
import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## function to compute the standard gaussian N(x;0,I) and a gaussian parametrized by
## mean mu and variance sigma log N(x|µ,σ)
# def log_gaussian(x, mu, log_var):
#     """
#     Returns the log pdf of a normal distribution parametrised
#     by mu and log_var evaluated at x. (Univariate distribution)
#     :param x: point to evaluate
#     :param mu: mean of distribution
#     :param log_var: log variance of distribution
#     :return: log N(x|µ,σ)
#     """
#     log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
#     return torch.sum(log_pdf, dim=-1)
#
#
# a = torch.Tensor([[2],[2],[2],[2]])
# mu = torch.Tensor([[1],[1],[1],[4]])
# log_var = torch.Tensor([[2],[2],[2],[1]])
#
# print(log_gaussian(a,mu,log_var))
def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x. (Univariate distribution)
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


def _kl_divergence(z, q_params, p_params=None):
    '''
    The function compute the KL divergence between the distribution q_phi(z|x) and the prior p_theta(z)
    of a sample z.

    KL(q_phi(z|x) || p_theta(z))  = -∫ q_phi(z|x) log [ p_theta(z) / q_phi(z|x) ]
                                  = -E[log p_theta(z) - log q_phi(z|x)]

    :param z: sample from the distribution q_phi(z|x)
    :param q_params: (mu, log_var) of the q_phi(z|x)
    :param p_params: (mu, log_var) of the p_theta(z)
    :return: the kl divergence KL(q_phi(z|x) || p_theta(z)) computed in z
    '''

    ## we have to compute the pdf of z wrt q_phi(z|x)
    (mu, log_var) = q_params
    qz = log_gaussian(z, mu, log_var)
    print('qz: ', qz)
    ## we should do the same with p
    if p_params is None:
        pz = log_standard_gaussian(z)
    else:
        (mu, log_var) = p_params
        pz = log_gaussian(z, mu, log_var)
    print('pz: ', pz)
    kl = qz - pz
    print(kl.shape)
    return kl


a = torch.Tensor([[1],[2],[3],[4]])
b = torch.Tensor([[2],[2],[2],[2]])
q_params = (a,b)
z = torch.Tensor([[5],[5],[5],[5]])

kl = _kl_divergence(z, q_params)
print(kl)