''' Now that we have trained a VAE model, we want to evalueate it in some way.
    We estimate the probability of data under the model using an importance sampling technique.
    We can write the marginal likelihood of a datapoint as:
            log p_theta(x) = log E_q [p_theta(x,z) / q_phi(z|x)]
                           ~ log 1/L sum( (p_theta(x|z) * p(z)) / q_phi(z|x) )
'''

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
from VAE_personal_implementation.utils.code_to_load_the_dataset import load_MNIST_dataset

import matplotlib.pyplot as plt

def show_images(images, title=None, path=None):
    images = utils.make_grid(images)
    show_image(images[0], title, path)

def show_image(img, title = "", path = None):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()

use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

HIDDEN_LAYERS = [100]
Z_DIM = 3
BATCH_SIZE = 128

ORIGINAL_BINARIZED_MNIST = True
ESTIMATION_SAMPLES = 100

PATH = 'saved_models/VAE_zdim_3_epoch_200_elbo_-137.06693204101563_learnrate_0.0003_Andrea'

## we have the binarized MNIST
## in this case we look at the test set, since we are interested in these examples that
## were not used to train the model
test_set = datasets.MNIST('../MNIST_dataset', train=False, download=True,
                   transform=transforms.ToTensor())
print('Number of examples in the training set:', len(test_set))
print('Size of the image:', test_set[0][0].shape)
## we plot an example only to check it
idx_ex = 1000
x, y = test_set[idx_ex] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.title('Example n {}, label: {}'.format(idx_ex, y))
plt.show()

### we only check if it is binarized
input_dim = x.numpy().size
print('Size of the image:', input_dim)

if ORIGINAL_BINARIZED_MNIST:
    ## we load the original dataset by Larochelle
    train_loader, val_loader, test_loader = load_MNIST_dataset('Original_MNIST_binarized/', BATCH_SIZE, True, True,
                                                               True)
else:
    # we have the binarized MNIST
    ## TRAIN SET

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    ## TEST SET
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_dataset', train=False, transform=flatten_bernoulli),
    batch_size=BATCH_SIZE, shuffle=True)




## we can create our model and try to train it
model = VariationalAutoencoder(28*28, HIDDEN_LAYERS, Z_DIM)
print('Model overview and recap\n')
print(model)
print('\n')

# now we have to load the trained model dict
model.load_state_dict(torch.load(PATH))



## now for each datapoint of the test set we want to compute the marginal likelihood

marginal_log_likelihood = 0
model.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        if ORIGINAL_BINARIZED_MNIST:
            images = data
        else:
            images, labels = data
        images = images.to(device)

        batch_log_likelihood = torch.zeros((len(images), ESTIMATION_SAMPLES))

        for j in range(ESTIMATION_SAMPLES):
            # I have to forward the images through the model, this way we get the reconstruction
            reconstruction = model(images)
            # we should get the kl
            # kl = torch.sum(model.kl_divergence)
            kl = model.kl_analytical ## BATCH_SIZE elements
            # print(kl.shape)

            likelihood = - torch.sum(F.binary_cross_entropy(reconstruction, images, reduction = 'none'), 1) ## BATCH_SIZE element

            batch_log_likelihood[:,j] = likelihood - kl

        ## at the end we have this matrix of size BATCH_SIZE x ESTIMATION_SAMPLES
        # print(batch_log_likelihood)
        log_likel = math.log(1/ESTIMATION_SAMPLES) + torch.logsumexp(batch_log_likelihood, dim = 1)
        marginal_log_likelihood += torch.sum(log_likel)

print('The marginal log likelihood we get on average on a test example is:', marginal_log_likelihood / len(test_loader.dataset))



