import math
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from VAE_personal_implementation.VAE import VariationalAutoencoder
from sklearn.decomposition import PCA

a = np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
a = torch.from_numpy(a)

b = torch.logsumexp(a, dim=1)
print(b)