from models import *
import torch
import torch.nn as nn
import math
from torch.distributions import OneHotCategorical, Normal
import tqdm
import ot

use_cuda=torch.cuda.is_available()

def generate_categorical(batch_size, classes=5):
    logits = torch.randint(5, size=(batch_size, classes)).float()
    n_samples = torch.randint(100,150,(1,))
    dist = OneHotCategorical(logits=logits)
    samples=dist.sample(n_samples).transpose(0,1)
    return [samples.float()]

def generate_gaussian_1d(batch_size, return_params=False):
    mus, sigmas = (1+5*torch.rand(size=(batch_size, 2))).chunk(2, dim=-1)
    n_samples = torch.randint(100,150,(1,))
    dist = Normal(mus, sigmas)
    samples=dist.sample(n_samples).transpose(0,1)
    if not return_params:
        return [samples.float()]
    else:
        return [samples.float()], (mus, sigmas)

def generate_multi(fct):
    def generate(*args, **kwargs):
        return fct(*args, **kwargs)[0], fct(*args, **kwargs)[0]
    return generate

def generate_multi_params(fct):
    def generate(*args):
        (X,), T_x = fct(*args, return_params=True)
        (Y,), T_y = fct(*args, return_params=True)
        return (X,Y), (T_x, T_y)
    return generate

def mode(samples):
    return samples.sum(dim=1).argmax(dim=-1)

def entropy(samples):
    counts = samples.sum(dim=-2) + 1
    probs = counts / counts.sum(dim=-1, keepdim=True)
    return -1 * (probs * torch.log(probs)).sum(dim=-1)

def KL_categorical(X, Y):
    counts_X = X.sum(dim=-2) + 1
    counts_Y = Y.sum(dim=-2) + 1
    probs_X = counts_X / counts_X.sum(dim=-1, keepdim=True)
    probs_Y = counts_Y / counts_Y.sum(dim=-1, keepdim=True)
    return (probs_X * (torch.log(probs_X) - torch.log(probs_Y))).sum(dim=-1)

def knn(X, k, Y=None, bs=32):
    if Y is None:
        Y = X
        k += 1
    X = X if type(X) == torch.Tensor else torch.Tensor(X)
    Y = Y if type(Y) == torch.Tensor else torch.Tensor(Y)
    outer_bs = Y.size(0)
    N = Y.size(1)
    n_batches = int(math.ceil(N/bs))
    dists = torch.zeros(outer_bs,N)
    if torch.cuda.is_available():
        X = X.to('cuda')
        Y = Y.to('cuda')
        dists=dists.to('cuda')
    for i in range(n_batches):
        j_min = i*bs
        j_max = min(N, (i+1)*bs)
        all_dists_i = (Y[:,j_min:j_max].unsqueeze(2) - X.unsqueeze(1)).norm(dim=-1)
        topk_i = all_dists_i.topk(k, dim=-1, largest=False)[0][:,:,k-1]
        dists[:,j_min:j_max] = topk_i
    return dists

def kl_knn(X, Y, k=1, xi=1e-5):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(-1)

    nu = knn(X=Y, Y=X, k=k) + xi
    eps = knn(X=X, k=k) + xi

    return d/n * torch.log(nu/eps).sum(dim=1) + math.log(m/(n-1))

def kl_knn_simple(X, Y, k=1, xi=1e-5):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(-1)

    nu = knn(X=Y, Y=X, k=k) + xi
    eps = knn(X=X, k=k) + xi

    return torch.log(nu/eps).sum(dim=1)/n

def entropy_1d_gaussian(mu, sigma):
    return torch.log(sigma)+1./2*math.log(2*math.pi) + 1./2

def kl_1d_gaussian(mu1, sigma1, mu2, sigma2):
    return torch.log(sigma2/sigma1) + (sigma1*sigma1 + (mu1-mu2)*(mu1-mu2))/2/sigma2/sigma2 - 1./2

def avg_nn_dist(X):
    dists = knn(X, 1)
    return dists.sum(dim=-1)/dists.size(-1)

def avg_log_nn_dist(X, xi=1e-5):
    dists = knn(X, 1)
    return torch.log(dists + xi).sum(dim=-1)/dists.size(-1)

def wasserstein(X, Y):
    costs = ot.dist(X, Y)
    return ot.emd2([],[],costs)

def train(model, sample_fct, label_fct, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in tqdm.tqdm(range(steps)):
        if exact_loss:
            X, theta = sample_fct(batch_size, return_params=True)
            if use_cuda:
                X = [x.cuda() for x in X]
                theta = [t.cuda() for t in theta]
            labels = label_fct(*theta)
        else:
            X = sample_fct(batch_size)
            if use_cuda:
                X = [x.cuda() for x in X]
            labels = label_fct(*X)
        loss = criterion(model(*X).squeeze(-1), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def show_examples(model, sample_fct, label_fct, n=8):
    X = sample_fct(n)
    if use_cuda:
        X = [x.cuda() for x in X]
    yhat = model(*X).cpu().detach()
    y = label_fct(*X).cpu()
    print("Y:", y, "\nYhat:", yhat)