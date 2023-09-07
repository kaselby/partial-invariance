import torch
import torch.nn as nn
import torch.nn.functional as F
import math

use_cuda=torch.cuda.is_available()



def batched_shuffle(x, dim=1):
    indices = torch.argsort(torch.rand(*x.size()[:2]), dim=dim)
    #result = torch.gather(x, dim, indices)
    result = x[torch.arange(x.shape[0]).unsqueeze(-1), indices]
    return result

def batched_cov(X, Y=None):
    # X bs x n x d
    N = X.size(1) - 1
    Xbar = X.sum(dim=1, keepdim=True) / N
    Xcentered = X - Xbar 
    if Y is not None:
        assert Y.size(1) == X.size(1)
        Ybar = Y.sum(dim=1, keepdim=True) / N
        Ycentered = Y - Ybar 
    else:
        Ycentered = Xcentered
    cov = Xcentered.transpose(1,2).matmul(Ycentered) / N
    return cov


def masked_softmax(x, mask, dim=-1, eps=1e-8):
    x_masked = x.clone()
    x_masked = x_masked - x_masked.max(dim=dim, keepdim=True)[0]
    x_masked[mask == 0] = -float("inf")
    return torch.exp(x_masked) / (torch.exp(x_masked).sum(dim=dim, keepdim=True) + eps)

def generate_masks(X_lengths, Y_lengths):
    X_max, Y_max = max(X_lengths), max(Y_lengths)

    X_mask = torch.arange(X_max)[None, :] < X_lengths[:, None]
    Y_mask = torch.arange(Y_max)[None, :] < Y_lengths[:, None]

    if use_cuda:
        X_mask = X_mask.cuda()
        Y_mask = Y_mask.cuda()

    mask_xx = X_mask.long()[:,:,None].matmul(X_mask.long()[:,:,None].transpose(1,2))
    mask_yy = Y_mask.long()[:,:,None].matmul(Y_mask.long()[:,:,None].transpose(1,2))
    mask_xy = X_mask.long()[:,:,None].matmul(Y_mask.long()[:,:,None].transpose(1,2))
    mask_yx = Y_mask.long()[:,:,None].matmul(X_mask.long()[:,:,None].transpose(1,2))

    return mask_xx, mask_xy, mask_yx, mask_yy

def poisson_loss(outputs, targets):
    return -1 * (targets * outputs - torch.exp(outputs)).mean()

def linear_block(input_size, hidden_size, output_size, num_layers, activation_fct=nn.ReLU):
    if num_layers == 0:
        layers = [nn.Linear(input_size, hidden_size), activation_fct(), nn.Linear(hidden_size, output_size)]
    elif num_layers > 0:
        layers = [nn.Linear(input_size, hidden_size), activation_fct()]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
    else:
        raise AssertionError("num_layers must be >= 0")
    return nn.Sequential(*layers)


def whiten(X):
    mu = X.mean(dim=1, keepdim=True)
    Sigma2 = (X-mu).transpose(1,2).matmul((X-mu)) / (X.size(1)-1)
    evals, evecs = torch.linalg.eig(Sigma2)
    lambd = (evecs.matmul(torch.diag_embed(evals.pow(-1./2))).matmul(evecs.transpose(1,2))).real
    return (X-mu).matmul(lambd.transpose(1,2))

def whiten_split(X,Y):
    n = X.size(1)
    D = torch.cat([X,Y],dim=1)
    Dp = whiten(D)
    return Dp[:, :n], Dp[:, n:]

def normalize_sets(*X):
    avg_norm = torch.cat(X, dim=1).norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True)
    return [x / avg_norm for x in X], avg_norm


# distance functions

def knn_dist(X, k, Y=None, bs=32):
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

    nu = knn_dist(X=Y, Y=X, k=k) + xi
    eps = knn_dist(X=X, k=k) + xi

    return d/n * torch.log(nu/eps).sum(dim=1) + math.log(m/(n-1))

def kl_mc(p, q, X=None, Y=None, N=500):
    if X is None:
        X = p.sample((N,)).transpose(0,1)
    return (p.log_prob(X.transpose(0,1)) - q.log_prob(X.transpose(0,1))).mean(dim=0)  

def kl_mc_mixture(p, q, X=None, Y=None, N=500):
    if X is None:
        X = p.sample((N,))
    else:
        X = [x.transpose(0,1) for x in X]
    X = [x.squeeze(-1) for x in X]
    return (p.log_prob(*X) - q.log_prob(*X)).mean(dim=0)  


def mi_corr_gaussian(corr, d=None, X=None):
    assert (d is None) != (X is None)
    if X is not None:
        d = X.size(-1)
    return -d/2 * torch.log(1-torch.pow(corr, 2))


def get_dists(X, Y=None, bs=32):
    if Y is None:
        Y = X
    X = X if type(X) == torch.Tensor else torch.Tensor(X)
    Y = Y if type(Y) == torch.Tensor else torch.Tensor(Y)
    outer_bs = Y.size(0)
    N = X.size(1)
    M = Y.size(1)
    n_batches = int(math.ceil(N/bs))
    dists = torch.zeros(outer_bs, N, M)
    if torch.cuda.is_available():
        X = X.to('cuda')
        Y = Y.to('cuda')
        dists=dists.to('cuda')
    for i in range(n_batches):
        j_min = i*bs
        j_max = min(N, (i+1)*bs)
        all_dists_i = (X[:,j_min:j_max].unsqueeze(2) - Y.unsqueeze(1)).norm(dim=-1)
        dists[:,j_min:j_max] = all_dists_i
    return dists

def kraskov_mi1(X, Y, k=1):
    assert X.size(1) == Y.size(1)
    N = X.size(1)
    d = X.size(-1)
    mask = (torch.eye(N)).to(X.device)
    mask[mask==1] = float('inf')
    Xdists = get_dists(X, X) + mask
    Ydists = get_dists(Y, Y) + mask
    Zdists = torch.maximum(Xdists, Ydists)
    eps,_ = Zdists.topk(k, dim=-1, largest=False)
    n_x = (Xdists < eps).float().sum(dim=-1)
    n_y = (Ydists < eps).float().sum(dim=-1)
    out = torch.digamma(torch.tensor([k], device=X.device)) + torch.digamma(torch.tensor([N], device=X.device)) - (torch.digamma(n_x+1) + torch.digamma(n_y+1)).mean(dim=1)
    return out

def kraskov_mi2(X, Y, k=1):
    assert X.size(1) == Y.size(1)
    N = X.size(1)
    d = X.size(-1)
    mask = (torch.eye(N)).to(X.device)
    mask[mask==1] = float('inf')
    Xdists,_ = torch.sort(get_dists(X, X) + mask, dim=-1)
    Ydists,_ = torch.sort(get_dists(Y, Y) + mask, dim=-1)
    eps_x = Xdists[:,:,k]
    eps_y = Ydists[:,:,k]
    eps = torch.maximum(eps_x, eps_y)
    n_x = (Xdists < eps_x.unsqueeze(-1)).float().sum(dim=-1)
    n_y = (Ydists < eps_y.unsqueeze(-1)).float().sum(dim=-1)

    out = torch.digamma(k) + torch.digamma(N) - (torch.digamma(n_x) + torch.digamma(n_y).mean(dim=1)) - 1/k
    return out


