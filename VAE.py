'''
In this file we are going to create our first implementation of a VAE, following
the Kingma and Welling [2014] paper. I cannot be sure it will be an optimized version,
but I will try to do my best.


'''
import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import numpy as np

## function to compute the standard gaussian N(x;0,I) and a gaussian parametrized by
## mean mu and variance sigma log N(x|µ,σ)
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
    :return: log N(x|mu,var)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    # print('Size log_pdf:', log_pdf.shape)
    return torch.sum(log_pdf, dim=-1)

## in a simple explanation a VAE is made up of three different parts:
## - Inference model (or encoder) q_phi(z|x)
## - A stochastic layer that sample (Reparametrization trick)
## - a generative model (or decoder) p_theta(z|x)
## given this, we want to minimize the ELBO

def reparametrization_trick(mu, log_var):
    '''
    Function that given the mean (mu) and the logarithmic variance (log_var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample

    :param mu: mean of the z_variables
    :param log_var: variance of the latent variables
    :return: z = mu + sigma * noise
    '''
    # we should get the std from the log_var
    # log_std = 0.5 * log_var (use the logarithm properties)
    # std = exp(log_std)
    std = torch.exp(log_var * 0.5)

    # we have to sample the noise (we do not have to keep the gradient wrt the noise)
    eps = Variable(torch.randn_like(std), requires_grad=False)
    z = mu.addcmul(std, eps)

    return z


## TODO: we can modify the method after to make it able to accept different sample layers
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        '''
        Probabilistic inference network given by a MLP. In case of a Gaussian MLP, we will
        have to output: log(sigma^2) and mu.

        :param input_dim: dimension of the input (scalar)
        :param hidden_dims: dimensions of the hidden layers (vector)
        :param latent_dim: dimension of the latent space
        '''

        super(Encoder, self).__init__()

        ## now we have to create the architecture
        neurons = [input_dim, *hidden_dims]
        ## common part of the architecture
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1,len(neurons))])

        ## we have two output: mu and log(sigma^2) #TODO: we can create a specific gaussian layer
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)


    def forward(self, input):
        x = input
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        ## now we should compute the mu and log var
        _mu = self.mu(x)
        # _log_var = F.softplus(self.log_var(x))
        _log_var = self.log_var(x)

        ## now we have also to return our z as the reparametrization trick told us
        ## z = mu + sigma * noise, where the noise is sample

        z = reparametrization_trick(_mu, _log_var)

        return z, _mu, _log_var


## now we have to create the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, input_dim):
        '''

        :param latent_dim: dimension of the latent space (scalar)
        :param hidden_dims: dimensions of the hidden layers (vector)
        :param input_dim: dimension of the input (scalar)
        '''

        super(Decoder, self).__init__()

        # this is kind of symmetric to the encoder, it starts from the latent variables z and it
        # tries to get the original x back

        neurons = [latent_dim, *hidden_dims]
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])

        self.reconstruction = nn.Linear(hidden_dims[-1], input_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        x = input
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # print(self.test_set_reconstruction(x).shape)
        return self.output_activation(self.reconstruction(x))


## at this point we have both the encoder and decoder, so we can create the VAE

class VariationalAutoencoder(nn.Module):
    def __init__(self,  input_dim, hidden_dims, latent_dim):
        '''
        Variational AutoEncoder as described in Kingma and Welling 2014. We have an encoder - decoder
        and we want to learn a meaningful latent representation to being able to reconstruct the input

        :param input_dim: dimension of the input
        :param hidden_dims: dimension of hidden layers #todo: maybe we can differentiate between the encoder and decoder?
        :param latent_dim: dimension of the latent variables
        '''

        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.z_dims = latent_dim

        ## we should create the encoder and the decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)
        self.kl_divergence = 0

        ## we should initialize the weights #TODO: INITIALIZE THE WEIGHTS as Kingma paper N(0,0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kl_divergence(self, z, q_params, p_params = None):
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
        # print('size qz:', qz.shape)
        ## we should do the same with p
        if p_params is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_params
            pz = log_gaussian(z, mu, log_var)
            # print('size pz:', pz.shape)

        kl = qz - pz

        return kl

    ## in case we are using a gaussian prior and a gaussian approximation family
    def _analytical_kl_gaussian(self, q_params):
        '''
        Way for computing the kl in an analytical way. This works for gaussian prior
        and gaussian density family for the approximated posterior.

        :param q_params: (mu, log_var) of the q_phi(z|x)
        :return: the kl value computed analytically
        '''

        (mu, log_var) = q_params
        # print(mu.shape)
        # print(log_var.shape)
        # prova = (log_var + 1 - mu**2 - log_var.exp())
        # print(prova.shape)
        # print(torch.sum(prova, 1).shape)
        # kl = 0.5 * torch.sum(log_var + 1 - mu**2 - log_var.exp(), 1)
        kl = 0.5 * torch.sum(log_var + 1 - mu.pow(2) - log_var.exp(), 1)

        return kl




    def forward(self, input):
        '''
        Given an input, we want to run the encoder, compute the kl, and reconstruct it

        :param input: an input example
        :return: for each pixel it returns the mean of the distribution of the values of that pixel
        '''

        # we pass the input through the encoder
        z, z_mu, z_log_var = self.encoder(input)
        # print(z.shape)

        # we compute the kl
        self.kl_divergence = self._kl_divergence(z, (z_mu, z_log_var))
        self.kl_analytical = self._analytical_kl_gaussian((z_mu, z_log_var))


        # we reconstruct it
        #         print(z
        x_mu = self.decoder(z)

        return x_mu


    def sample(self, n_images):
        '''
        Method to sample from our generative model

        :return: a sample starting from z ~ N(0,1)
        '''

        z = torch.randn((n_images, self.z_dims), dtype = torch.float)
        # print(z)
        samples =  self.decoder(z)

        return samples
















