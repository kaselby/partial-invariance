from torch.distributions.distribution import Distribution
#from models import *
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.distributions import OneHotCategorical, Normal, MultivariateNormal, Categorical, MixtureSameFamily, Distribution, Dirichlet, LKJCholesky, LogNormal
import tqdm
import ot
from geomloss import SamplesLoss

use_cuda=torch.cuda.is_available()


def windowed_losses(losses, window_size, stride):
    


def generate_categorical(batch_size, classes=5):
    logits = torch.randint(5, size=(batch_size, classes)).float()
    n_samples = torch.randint(100,150,(1,))
    dist = OneHotCategorical(logits=logits)
    samples=dist.sample(n_samples).transpose(0,1)
    return [samples.float()]

def generate_gaussian_1d(batch_size, return_params=False, set_size=(100,150)):
    mus, sigmas = (1+5*torch.rand(size=(batch_size, 2))).chunk(2, dim=-1)
    n_samples = torch.randint(*set_size,(1,))
    dist = Normal(mus, sigmas)
    samples=dist.sample(n_samples).transpose(0,1)
    if not return_params:
        return [samples.float().contiguous()]
    else:
        return [samples.float().contiguous()], (mus, sigmas)

def generate_gaussian_nd(batch_size, n, return_params=False, set_size=(100,150), scale=None):
    mus= torch.rand(size=(batch_size, n))
    A = torch.rand(size=(batch_size, n, n)) 
    if scale is not None:
        mus = mus * scale.unsqueeze(-1).unsqueeze(-1)
        A = A * torch.sqrt(scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) 
    else:
        mus= 1+5*mus
        A = A
    sigmas = torch.bmm(A.transpose(1,2), A) + 1*torch.diag_embed(torch.rand(batch_size, n))
    n_samples = torch.randint(*set_size,(1,))
    dist = MultivariateNormal(mus, sigmas)
    samples=dist.sample(n_samples).transpose(0,1)
    if return_params:
        return [samples.float().contiguous()], [dist]
    else:
        return [samples.float().contiguous()]
        

def generate_gaussian_variable_dim(batch_size, dims=(2,6), **kwargs):
    n = torch.randint(*dims,(1,)).item()
    return generate_gaussian_nd(batch_size, n, **kwargs)

def generate_gaussian_variable_dim_multi(batch_size, dims=(2,6), **kwargs):
    n = torch.randint(*dims,(1,)).item()
    X = generate_gaussian_nd(batch_size, n, **kwargs)
    Y = generate_gaussian_nd(batch_size, n, **kwargs)
    return X[0],Y[0]

def generate_gaussian_mixture_variable_dim_multi(batch_size, dims=(2,6), normalize=False, scaleinv=False, **kwargs):
    n = torch.randint(*dims,(1,)).item()
    scale = torch.exp(torch.rand(batch_size)*9 - 6) if scaleinv else None
    X = generate_gaussian_mixture(batch_size, n, scale=scale, **kwargs)[0]
    Y = generate_gaussian_mixture(batch_size, n, scale=scale, **kwargs)[0]
    if normalize:
        avg_norm = torch.cat([X,Y], dim=1).norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True)
        X /= avg_norm
        Y /= avg_norm
    return X, Y

def generate_uniform(batch_size, eps=1e-4):
    n_samples = torch.randint(100,150,(1,))
    samples = torch.rand(size=(batch_size, n_samples)) * (1-2*eps) + eps
    return [samples.float().contiguous()]

def generate_uniform_nd_scaleinv(batch_size, n):
    scale = torch.exp(torch.random()*3)
    n_samples = torch.randint(100,150,(1,))
    samples = torch.rand(size=(batch_size, n_samples, n)) * scale
    return [samples.float().contiguous()]

def generate_uniform_nd(batch_size, n):
    n_samples = torch.randint(100,150,(1,))
    samples = torch.rand(size=(batch_size, n_samples, n))
    return [samples.float().contiguous()]

def generate_gaussian_mixture(batch_size, n, return_params=False, set_size=(100,150), component_range=(3,10), scale=None):
    n_samples = torch.randint(*set_size,(1,))
    n_components = torch.randint(*component_range,(1,))
    mus= torch.rand(size=(batch_size, n_components, n))
    A = torch.rand(size=(batch_size, n_components, n, n)) 
    if scale is not None:
        mus = mus * scale.unsqueeze(-1).unsqueeze(-1)
        A = A * torch.sqrt(scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) 
    else:
        mus= 1+5*mus
        A = A
    sigmas = A.transpose(2,3).matmul(A) + torch.diag_embed(torch.rand(batch_size, n_components, n))
    logits = torch.randint(5, size=(batch_size, n_components)).float()
    base_dist = MultivariateNormal(mus, sigmas)
    mixing_dist = Categorical(logits=logits)
    dist = MixtureSameFamily(mixing_dist, base_dist)
    samples = dist.sample(n_samples).transpose(0,1)
    if return_params:
        return [samples.float().contiguous()], [dist]
    else:
        return [samples.float().contiguous()]


def generate():
    scale = torch.exp(torch.rand(8)*9 - 6)
    X,Y = generate_gaussian_mixture(8, scale=scale, n=32)[0], generate_gaussian_mixture(8, scale=scale, n=32)[0]
    avg_norm = torch.cat([X,Y], dim=1).norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True)
    X /= avg_norm
    Y /= avg_norm
    print(scale)
    print(wasserstein_exact(X,Y))

'''
def generate_gaussian_mixture(batch_size, n, return_params=False):
    n_samples = torch.randint(100,150,(1,))
    n_components = torch.randint(3,10,(1,))
    mus= (1+5*torch.rand(size=(batch_size, n_components, n)))
    A = torch.rand(size=(batch_size, n_components, n, n))
    sigmas = A.transpose(-1,-2).matmul(A) + 1*torch.diag_embed(torch.rand(batch_size, n_components, n))
    logits = torch.randint(5, size=(batch_size, n_components)).float()
    mix = Categorical(logits=logits)
    dist = MultivariateNormal(mus, sigmas)
    indices = mix.sample(n_samples).transpose(0,1)
    samples = dist.sample(n_samples).transpose(0,1)
    out = torch.gather(samples, 2, indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,n)).squeeze(2)
    if not return_params:
        return [out.float().contiguous()]
    else:
        return [out.float().contiguous()], (mus, sigmas, mix.probs)
'''



def generate_multi(fct, normalize=False, scaleinv=False):
    def generate(*args, **kwargs):
        scale = torch.exp(torch.rand(1)*9 - 6) if scaleinv else None
        X,Y = fct(*args, scale=scale, **kwargs)[0], fct(*args, scale=scale, **kwargs)[0]
        if normalize:
            avg_norm = torch.cat([X,Y], dim=1).norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True)
            X /= avg_norm
            Y /= avg_norm
        return X,Y
    return generate

def generate_multi_params(fct):
    def generate(*args, **kwargs):
        (X,), T_x = fct(*args, return_params=True, **kwargs)
        (Y,), T_y = fct(*args, return_params=True, **kwargs)
        return (X,Y), (*T_x, *T_y)
    return generate

def join(f1, f2):
    def f(*args, **kwargs):
        return f1(*args, **kwargs)[0], f2(*args, **kwargs)[0]
    return f

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

def knn_inds(X, k, Y=None, bs=32):
    if Y is None:
        Y = X
        exclude_0 = True
    else:
        exclude_0 = False
    X = X if type(X) == torch.Tensor else torch.Tensor(X)
    Y = Y if type(Y) == torch.Tensor else torch.Tensor(Y)
    outer_bs = Y.size(0)
    N = Y.size(1)
    n_batches = int(math.ceil(N/bs))
    inds = torch.zeros(outer_bs,N,k, dtype=torch.int64)
    if torch.cuda.is_available():
        X = X.to('cuda')
        Y = Y.to('cuda')
        inds=inds.to('cuda')
    for i in range(n_batches):
        j_min = i*bs
        j_max = min(N, (i+1)*bs)
        all_inds_i = (Y[:,j_min:j_max].unsqueeze(2) - X.unsqueeze(1)).norm(dim=-1)
        topk_i = all_inds_i.topk(k+1, dim=-1, largest=False)[1]
        if exclude_0:
            inds[:,j_min:j_max, :] = topk_i[:,:,1:k+1]
        else:
            inds[:,j_min:j_max, :] = topk_i[:,:,:k]
    return inds

def cross_knn_inds(X, Y, k):
    N_XX = knn_inds(X, k)
    N_YY = knn_inds(Y, k)
    N_XY = knn_inds(Y, k, X)
    N_YX = knn_inds(X, k, Y)
    return N_XX, N_XY, N_YX, N_YY

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


def kl_knn2(X, Y):
    N = X.size(1)
    M = Y.size(1)
    d = X.size(-1)
    Xmask = (torch.eye(N)).to(X.device)
    Xmask[Xmask==1] = float('inf')
    Xdists,_ = torch.sort(get_dists(X, X) + Xmask, dim=-1)
    Ydists,_ = torch.sort(get_dists(X, Y), dim=-1)
    rho = Xdists[:,:,0]
    nu = Ydists[:,:,0]
    eps = torch.maximum(rho, nu)
    l = (Xdists <= eps.unsqueeze(-1)).float().sum(dim=-1) + 1
    k = (Ydists <= eps.unsqueeze(-1)).float().sum(dim=-1)

    kl1 = (torch.digamma(l) - torch.digamma(k)).mean(dim=-1) + math.log(M/(N-1))
    #return kl
    rho_l = torch.gather(Xdists, -1, l.long().unsqueeze(-1)).squeeze(-1)
    nu_k = torch.gather(Ydists, -1, k.long().unsqueeze(-1)).squeeze(-1)
    kl2 = torch.log(nu_k/rho_l).mean(dim=-1) * d + kl1
    return kl1, kl2



def simplified_divergence1(X, Y, k=1, xi=1e-5):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(-1)

    nu = knn_dist(X=Y, Y=X, k=k) + xi
    eps = knn_dist(X=X, k=k) + xi

    return 1/n * (torch.log(nu) + torch.log(eps)).sum(dim=1)

def simplified_divergence2(X, Y, k=1, xi=1e-5):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(-1)

    nu = knn_dist(X=Y, Y=X, k=k) + xi
    eps = knn_dist(X=X, k=k) + xi

    return 1/n * (torch.log(nu) - torch.log(eps)).sum(dim=1)

def kl_knn_simple(X, Y, k=1, xi=1e-5):
    n = X.size(1)
    m = Y.size(1)
    d = X.size(-1)

    nu = knn_dist(X=Y, Y=X, k=k) + xi
    eps = knn_dist(X=X, k=k) + xi

    return torch.log(nu/eps).sum(dim=1)/n

def entropy_1d_gaussian(mu, sigma):
    return torch.log(sigma).squeeze(-1)+1./2*math.log(2*math.pi) + 1./2

def kl_1d_gaussian(mu1, sigma1, mu2, sigma2):
    return torch.log(sigma2/sigma1) + (sigma1*sigma1 + (mu1-mu2)*(mu1-mu2))/2/sigma2/sigma2 - 1./2

def kl_nd_gaussian(P, Q):
    mu1, Sigma1 = P.loc, P.covariance_matrix
    mu2, Sigma2 = Q.loc, Q.covariance_matrix
    d = mu1.size(-1)
    Lambda2 = Q.precision_matrix
    return 1./2 * ( torch.logdet(Sigma2) - torch.logdet(Sigma1) -
                d +
                torch.diagonal(Lambda2.matmul(Sigma1), dim1=-2, dim2=-1).sum(dim=-1) +
                (mu2-mu1).unsqueeze(-1).transpose(-1, -2).matmul(Lambda2).matmul((mu2-mu1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )

def kl_nd_gaussian2(P, Q):
    mu1, Sigma1 = P.loc, P.covariance_matrix
    mu2, Sigma2 = Q.loc, Q.covariance_matrix
    d = mu1.size(-1)
    Lambda2 = Q.precision_matrix
    return (torch.logdet(Sigma2) - torch.logdet(Sigma1), 
                -d, 
                torch.diagonal(Lambda2.matmul(Sigma1), dim1=-2, dim2=-1).sum(dim=-1),
                (mu2-mu1).unsqueeze(-1).transpose(-1, -2).matmul(Lambda2).matmul((mu2-mu1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )

def kl_mc(p, q, X=None, Y=None, N=500):
    if X is None:
        X = p.sample((N,)).transpose(0,1)
    return (p.log_prob(X.transpose(0,1)) - q.log_prob(X.transpose(0,1))).mean(dim=0)    


# bs x nc (x d for mu and x d x d for Sigma)
def kl_gmm_bounds(mu1, Sigma1, w1, mu2, Sigma2, w2):
    d = mu1.size(-1)
    H = gaussian_nd_entropy(Sigma1)
    lower_term1 = weighted_logsumexp(kl_nd_gaussian(mu1.unsqueeze(2), Sigma1.unsqueeze(2), mu1.unsqueeze(1), Sigma1.unsqueeze(1)), w1.unsqueeze(1), dim=2) #dim=1 or 2?
    lower_term2 = weighted_logsumexp(gaussian_product_lognorm(mu1.unsqueeze(2), Sigma1.unsqueeze(2), mu2.unsqueeze(1), Sigma2.unsqueeze(1)), w2.unsqueeze(1), dim=2) #dim=1 or 2?
    lower_bound = (w1 * (lower_term1 - lower_term2 - H)).sum(dim=1) 

    upper_term1 = weighted_logsumexp(gaussian_product_lognorm(mu1.unsqueeze(2), Sigma1.unsqueeze(2), mu1.unsqueeze(1), Sigma1.unsqueeze(1)), w1.unsqueeze(1), dim=2) #dim=1 or 2?
    upper_term2 = weighted_logsumexp(kl_nd_gaussian(mu1.unsqueeze(2), Sigma1.unsqueeze(2), mu2.unsqueeze(1), Sigma2.unsqueeze(1)), w2.unsqueeze(1), dim=2) #dim=1 or 2?
    upper_bound = (w1 * (upper_term1 - upper_term2 + H)).sum(dim=1)

    return lower_bound, upper_bound

# logits and weights both (*, N)
# add eps?
def weighted_logsumexp(logits, weights, dim=-1):
    max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
    out = torch.log(torch.sum(weights * torch.exp(logits - max_logits), dim=dim)) + max_logits
    return out

def gaussian_product_lognorm(mu1, Sigma1, mu2, Sigma2):
    d = mu1.size(-1)
    return -1./2 * (d*math.log(2*math.pi) + torch.logdet(Sigma1 + Sigma2) + (mu2-mu1).unsqueeze(-2).matmul(torch.inverse(Sigma1+Sigma2)).matmul((mu2-mu1).unsqueeze(-1))).squeeze(-1).squeeze(-1)

def gaussian_nd_entropy(Sigma):
    d = Sigma.size(-1)
    return 1./2 * (d*math.log(2*math.pi) + d + torch.logdet(Sigma))

def avg_nn_dist(X):
    dists = knn_dist(X, 1)
    return dists.sum(dim=-1)/dists.size(-1)

def avg_cross_nn_dist(X, Y):
    dists = knn_dist(X=Y, Y=X, k=1)
    return dists.sum(dim=-1)/dists.size(-1)

def avg_log_nn_dist(X, xi=1e-5):
    dists = knn_dist(X, 1)
    return torch.log(dists + xi).sum(dim=-1)/dists.size(-1)

def avg_log_cross_nn_dist(X, Y, xi=1e-5):
    dists = knn_dist(X=Y, Y=X, k=1)
    return torch.log(dists+xi).sum(dim=-1)/dists.size(-1)

def wasserstein_exact(X, Y):
    bs = X.size(0)
    dists = torch.zeros(bs)
    for i in range(bs):
        costs = ot.dist(X[i].cpu().detach().numpy(), Y[i].cpu().detach().numpy(), metric='euclidean')
        dists[i] = ot.emd2([],[],costs)
    if use_cuda:
        dists = dists.cuda()
    return dists

def wasserstein2_exact(X, Y):
    bs = X.size(0)
    dists = torch.zeros(bs)
    for i in range(bs):
        costs = ot.dist(X[i].cpu().detach().numpy(), Y[i].cpu().detach().numpy(), metric='sqeuclidean')
        dists[i] = ot.emd2([],[],costs)
    if use_cuda:
        dists = dists.cuda()
    return dists.sqrt()

def wasserstein(X, Y, p=1, **kwargs):
    loss = SamplesLoss(p=p, **kwargs)
    out = loss(X, Y)
    if p > 1:
        out = (out*p).pow(1./p)
    return out

def wasserstein2(X, Y, **kwargs):
    loss = SamplesLoss(p=2, **kwargs)
    return (loss(X, Y)*2).sqrt()

def wasserstein2_gaussian(P, Q, X=None):
    mu1, Sigma1 = P.loc, P.covariance_matrix
    mu2, Sigma2 = Q.loc, Q.covariance_matrix
    s1 = matrix_pow(Sigma1, 1./2)
    return ((mu2-mu1).norm(dim=-1).pow(2) + torch.diagonal(Sigma1 + Sigma2 - 2 * matrix_pow(s1.matmul(Sigma2).matmul(s1), 1./2), dim1=-2, dim2=-1).sum(dim=-1)).sqrt()

def wasserstein_mc(P, Q, N=5000, X=None, **kwargs):
    X = P.sample((N,)).transpose(0,1).contiguous()
    Y = Q.sample((N,)).transpose(0,1).contiguous()
    return wasserstein(X, Y, **kwargs)

def mi_corr_gaussian(corr, d=None, X=None):
    assert (d is None) != (X is None)
    if X is not None:
        d = X.size(-1)
    return -d/2 * torch.log(1-torch.pow(corr, 2))


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




class PrecisionLoss(nn.Module):
    def forward(self, scores, labels):
        predictions = torch.sigmoid(scores)
        tp = predictions[labels==1].sum()
        return tp / predictions.sum()



'''
def train(model, sample_fct, label_fct, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, lr=1e-5, lr_decay=False, epoch_size=250, milestones=[], *sample_args, **sample_kwargs):
    #model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_decay:
        #isqrt = lambda step: 1/math.sqrt(step - warmup) if step > warmup else 1
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, isqrt)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    losses = []
    for i in tqdm.tqdm(range(1,steps+1)):
        optimizer.zero_grad()
        if exact_loss:
            X, theta = sample_fct(batch_size, *sample_args, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
                #theta = [t.cuda() for t in theta]
            labels = label_fct(*theta).squeeze(-1)
        else:
            X = sample_fct(batch_size, *sample_args, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
            labels = label_fct(*X)
        loss = criterion(model(*X).squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        #if lr_decay:
        #    scheduler.step()
        if lr_decay and i % epoch_size == 0:
            window_size = int(epoch_size / 10)
            windowed_avg= sum(losses[-window_size:])/window_size
            scheduler.step()

        losses.append(loss.item())
    return losses

def evaluate(model, sample_fct, label_fct, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, **sample_kwargs):
    #model.train(False)
    l1=nn.L1Loss()
    l2=nn.MSELoss()
    l1_losses=[]
    l2_losses=[]
    for _ in tqdm.tqdm(range(steps)):
        if exact_loss:
            X, theta = sample_fct(batch_size, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
                theta = [t.cuda() for t in theta]
            labels = label_fct(*theta).squeeze(-1)
        else:
            X = sample_fct(batch_size, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
            labels = label_fct(*X)
        l1loss = l1(model(*X).squeeze(-1), labels)
        l2loss = l2(model(*X).squeeze(-1), labels)
        l1_losses.append(l1loss.item())
        l2_losses.append(l2loss.item())
    return sum(l1_losses)/len(l1_losses), sum(l2_losses)/len(l2_losses)
'''

def eval_all(*models, sample_fct, label_fct, train_kwargs={}, eval_kwargs={}):
    for model in models:
        model_losses = train(model, sample_fct, label_fct, **train_kwargs)
        l1, l2 = evaluate(model, sample_fct, label_fct, **eval_kwargs)

        print("\nL1:", str(l1), "\tL2:", str(l2))

import tabulate
def show_examples(model, sample_fct, label_fct, exact_loss=False, samples=8, **sample_kwargs):
    #model.train(False)
    if exact_loss:
        X, theta = sample_fct(samples, **sample_kwargs)
        if use_cuda:
            X = [x.cuda() for x in X]
            theta = [t.cuda() for t in theta]
        y = label_fct(*theta).cpu().squeeze(-1)
    else:
        X = sample_fct(samples, **sample_kwargs)
        if use_cuda:
            X = [x.cuda() for x in X]
        y = label_fct(*X).cpu()
    yhat = model(*X).cpu().detach().squeeze(-1)
    print(tabulate.tabulate([['y', *y.tolist()], ['yhat', *yhat.tolist()]]))
    #print("Y:", y, "\nYhat:", yhat)

def show_samples(sample_fct,label_fct,samples=8,normalize=False,**kwargs):
    X = sample_fct(samples,**kwargs)
    if normalize:
        X = [X_i / X_i.norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True) for X_i in X]
    y = label_fct(*X)
    print(tabulate.tabulate([['y', *y.tolist()]]))

from copy import deepcopy
def scale_gmm(mixture, scale):
    scaled_dist = deepcopy(mixture)
    scaled_dist.component_distribution.loc *= scale
    scaled_dist.component_distribution.scale_tril *= scale
    return scaled_dist

from scipy.stats import invwishart
def sample_invwishart(df, scale, n):
    return invwishart.rvs(df, scale, size=n)


class GaussianGenerator():
    def __init__(self, num_outputs=1, mixture=True, normalize=False, scaleinv=False, return_params=False, variable_dim=False):
        self.num_outputs = num_outputs
        self.normalize = normalize
        self.scaleinv = scaleinv
        self.return_params = return_params
        self.variable_dim = variable_dim
        self.mixture = mixture
        self.device = torch.device('cpu') if not use_cuda else torch.device('cuda')

    def _generate(self, batch_size, n, return_params=False, set_size=(100,150), scale=None, nu=3, mu0=0, s0=1):
        n_samples = torch.randint(*set_size,(1,))
        mus= torch.rand(size=(batch_size, n))
        c = LKJCholesky(n, concentration=nu).sample((batch_size,))
        while c.isnan().any():
            c = LKJCholesky(n, concentration=nu).sample((batch_size,))
        #s = torch.diag_embed(LogNormal(mu0,s0).sample((batch_size, n)))
        sigmas = c#torch.matmul(s, c)
        if scale is not None:
            mus = mus * scale.unsqueeze(-1).unsqueeze(-1)
            sigmas = sigmas * scale.unsqueeze(-1).unsqueeze(-1)
        mus = mus.to(self.device)
        sigmas = sigmas.to(self.device)
        dist = MultivariateNormal(mus, scale_tril=sigmas)
        samples = dist.sample(n_samples).transpose(0,1)
        if return_params:
            return samples.float().contiguous(), dist
        else:
            return samples.float().contiguous()

    def _generate_mixture_old(self, batch_size, n, return_params=False, set_size=(100,150), component_range=(3,10), scale=None):
        n_samples = torch.randint(*set_size,(1,))
        n_components = torch.randint(*component_range,(1,))
        mus= torch.rand(size=(batch_size, n_components, n))
        A = torch.rand(size=(batch_size, n_components, n, n)) 
        if scale is not None:
            mus = mus * scale.unsqueeze(-1).unsqueeze(-1)
            A = A * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            mus= 1+5*mus
            A = A
        mus = mus.to(self.device)
        sigmas = (A.transpose(2,3).matmul(A) + torch.diag_embed(torch.rand(batch_size, n_components, n))).to(self.device)
        logits = torch.randint(5, size=(batch_size, n_components)).float().to(self.device)
        base_dist = MultivariateNormal(mus, sigmas)
        mixing_dist = Categorical(logits=logits)
        dist = MixtureSameFamily(mixing_dist, base_dist)
        samples = dist.sample(n_samples).transpose(0,1)
        if return_params:
            return samples.float().contiguous(), dist
        else:
            return samples.float().contiguous()

    def _generate_mixture(self, batch_size, n, return_params=False, set_size=(100,150), component_range=(1,10), scale=None, nu=5, mu0=0, s0=0.3):
        n_samples = torch.randint(*set_size,(1,))
        n_components = torch.randint(*component_range,(1,)).item()
        mus= torch.rand(size=(batch_size, n_components, n))
        c = LKJCholesky(n, concentration=nu).sample((batch_size, n_components))
        while c.isnan().any():
            c = LKJCholesky(n, concentration=nu).sample((batch_size, n_components))
        s = torch.diag_embed(LogNormal(mu0,s0).sample((batch_size, n_components, n)))
        sigmas = torch.matmul(s, c)
        if scale is not None:
            mus = mus * scale.unsqueeze(-1).unsqueeze(-1)
            sigmas = sigmas * scale.unsqueeze(-1).unsqueeze(-1)
        mus = mus.to(self.device)
        sigmas = sigmas.to(self.device)
        logits = Dirichlet(torch.ones(n_components).to(self.device)/n_components).sample((batch_size,))
        base_dist = MultivariateNormal(mus, scale_tril=sigmas)
        mixing_dist = Categorical(logits=logits)
        dist = MixtureSameFamily(mixing_dist, base_dist)
        samples = dist.sample(n_samples).transpose(0,1)
        if return_params:
            return samples.float().contiguous(), dist
        else:
            return samples.float().contiguous()
        

    
    def __call__(self, batch_size, dims=(2,6), **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        scale = torch.exp(torch.rand(batch_size)*9 - 6) if self.scaleinv else None
        gen_fct = self._generate_mixture if self.mixture else self._generate
        if self.return_params:
            outputs, dists = zip(*[gen_fct(batch_size, scale=scale, return_params=True, **kwargs) for _ in range(self.num_outputs)])
        else:
            outputs = [gen_fct(batch_size, scale=scale, **kwargs) for _ in range(self.num_outputs)]
        if self.normalize:
            avg_norm = torch.cat(outputs, dim=1).norm(dim=-1,keepdim=True).mean(dim=1,keepdim=True)
            for i in range(len(outputs)):
                outputs[i] /= avg_norm
            if self.return_params:
                for dist in dists:
                    dist.base_dist = MultivariateNormal(dist.loc/avg_norm, dist.covariance_matrix/avg_norm/avg_norm)
        if self.return_params:
            return outputs, dists
        else:
            return outputs

class PairedGaussianGenerator():
    def __init__(self, num_outputs=1, mixture=True, return_params=False, variable_dim=False):
        self.num_outputs = num_outputs
        self.return_params = return_params
        self.variable_dim = variable_dim
        self.mixture = mixture
        self.device = torch.device('cpu') if not use_cuda else torch.device('cuda')

    def _generate(self, batch_size, n, set_size=(100,150), component_range=(1,10), nu=1, mu0=0, s0=1, ddf=2):
        n_samples = torch.randint(*set_size,(1,))
        n_components = torch.randint(*component_range,(1,)).item()

        c = LKJCholesky(n, concentration=nu).sample()
        s = torch.diag_embed(LogNormal(mu0,s0).sample((n,)))
        scale_chol = torch.matmul(s, c)
        scale = scale_chol.matmul(scale_chol.t())

        def _generate_set():
            mus = MultivariateNormal(torch.zeros(n), scale_tril=scale_chol).sample((batch_size, n_components))
            sigmas = torch.tensor(invwishart.rvs(n+ddf, scale.numpy(), size=batch_size*n_components)).view(batch_size, n_components, n, n).float()
            mus, sigmas = mus.to(self.device), sigmas.to(self.device)
            logits = Dirichlet(torch.ones(n_components).to(self.device)/n_components).sample((batch_size,))
            base_dist = MultivariateNormal(mus, covariance_matrix=sigmas)
            mixing_dist = Categorical(logits=logits)
            dist = MixtureSameFamily(mixing_dist, base_dist)
            samples = dist.sample(n_samples).transpose(0,1)
            return dist, samples.float().contiguous()
        
        Xdist, X = _generate_set()
        Ydist, Y = _generate_set()

        if self.return_params:
            return (X, Y), (Xdist, Ydist)
        else:
            return X, Y
    
    def __call__(self, batch_size, dims=(2,6), **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        return self._generate(batch_size, **kwargs)


class CorrelatedGaussianGenerator():
    def __init__(self, return_params=False, variable_dim=False):
        self.return_params=return_params
        self.variable_dim=variable_dim
        
    def _build_dist(self, batch_size, corr, n):
        mu = torch.zeros((batch_size, n*2))
        I = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
        if use_cuda:
            I = I.cuda()
            mu = mu.cuda()
        rhoI = corr.unsqueeze(-1).unsqueeze(-1) * I
        cov = torch.cat([torch.cat([I, rhoI], dim=1), torch.cat([rhoI, I], dim=1)], dim=2)
        return MultivariateNormal(mu, covariance_matrix=cov)

    def _generate(self, batch_size, n, set_size=(100,150), corr=None):
        n_samples = torch.randint(*set_size,(1,))
        if corr is None:
            corr = 0.999-1.998*(torch.rand((batch_size,)))
            if use_cuda:
                corr = corr.cuda()
        dists = self._build_dist(batch_size, corr, n)
        X, Y = dists.sample(n_samples).transpose(0,1).chunk(2, dim=-1)
        if self.return_params:
            return (X, Y), (corr,)
        else:
            return X, Y

    def __call__(self, batch_size, dims=(2,6), **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        return self._generate(batch_size, **kwargs)



from flows import MADE, MADE_IAF, BatchNormFlow, Reverse, FlowSequential, BatchOfFlows
def build_maf(num_inputs, num_hidden, num_blocks, nf_cls=MADE_IAF):
    modules=[]
    for _ in range(num_blocks):
        modules += [
            nf_cls(num_inputs, num_hidden, None, act='relu'),
            #BatchNormFlow(num_inputs),
            Reverse(num_inputs)
        ]
    model = FlowSequential(*modules)
    model.num_inputs = num_inputs
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)
    if use_cuda:
        model = model.cuda()
    return model


class NFGenerator():
    def __init__(self, num_hidden, num_blocks, num_outputs=1, normalize=False, return_params=False, use_maf=False, variable_dim=False):
        self.num_hidden=num_hidden
        self.num_blocks=num_blocks

        self.num_outputs=num_outputs
        self.normalize=normalize
        self.return_params=return_params
        self.use_maf=use_maf
        self.variable_dim=variable_dim
        self.device = torch.device('cpu') if not use_cuda else torch.device('cuda')

    def _generate(self, batch_size, n, return_params=False, set_size=(100,150)):
        n_samples = torch.randint(*set_size,(1,))
        flows = BatchOfFlows(batch_size, n, self.num_hidden, self.num_blocks, use_maf=self.use_maf).to(self.device)
        samples = flows.sample(n_samples).transpose(0,1)
        if return_params:
            return samples, flows
        else: 
            return samples

    def _generate2(self, batch_size, n, return_params=False, set_size=(100,150)):
        n_samples = torch.randint(*set_size,(1,))
        nf_cls = MADE if self.use_maf else MADE_IAF
        mafs = [build_maf(n, self.num_hidden, self.num_blocks, nf_cls=nf_cls) for i in range(batch_size)]
        samples = torch.stack([x.sample(num_samples=n_samples) for x in mafs], dim=0)
        if return_params:
            return samples, mafs
        else: 
            return samples

    def __call__(self, batch_size, dims=(2,6), **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        if self.return_params:
            outputs, dists = zip(*[self._generate(batch_size, return_params=True, **kwargs) for _ in range(self.num_outputs)])
        else:
            outputs = [self._generate(batch_size, **kwargs) for _ in range(self.num_outputs)]
        if self.return_params:
            return outputs, dists
        else:
            return outputs


def test_gen1(*args,**kwargs):
    def generate(batch_size, n, set_size=(100,150),device=torch.device('cuda'), nu=1, mu=0, s=1):
        n_samples = torch.randint(*set_size,(1,))
        mus= torch.rand(size=(batch_size, n))
        c = LKJCholesky(n, concentration=nu).sample((batch_size,))
        while c.isnan().any():
            c = LKJCholesky(n, concentration=nu).sample((batch_size,))
        s = torch.diag_embed(LogNormal(mu,s).sample((batch_size, n)))
        sigmas = torch.matmul(s, c)
        mus = mus.to(device)
        sigmas = sigmas.to(device)
        dist = MultivariateNormal(mus, scale_tril=sigmas)
        samples = dist.sample(n_samples).transpose(0,1)
        return samples,dist
    X,Y = generate(*args, **kwargs), generate(*args, **kwargs)
    return zip(X,Y)

def test_gen2(*args,**kwargs):
    def generate(batch_size, n, set_size=(100,150),device=torch.device('cuda')):
        mus= 1+5*torch.rand(size=(batch_size, n))
        A = torch.rand(size=(batch_size, n, n)) 
        sigmas = torch.bmm(A.transpose(1,2), A) + 1*torch.diag_embed(torch.rand(batch_size, n))
        mus = mus.to(device)
        sigmas = sigmas.to(device)
        n_samples = torch.randint(*set_size,(1,))
        dist = MultivariateNormal(mus, sigmas)
        samples=dist.sample(n_samples).transpose(0,1)
        return samples.float().contiguous(), dist
    X,Y = generate(*args, **kwargs), generate(*args, **kwargs)
    return zip(X,Y)


def test_dist(d1, d2, n1, n2, N, k=1, bs=64):
    kl = 0
    for i in range(math.ceil(N/bs)):
        effective_bs = min(N, (i+1)*bs) - i*bs
        n_samples1 = torch.randint(n1,n2,(1,))
        n_samples2 = torch.randint(n1,n2,(1,))
        X,Y = d1.sample((effective_bs,n_samples1)).squeeze(2), d2.sample((effective_bs,n_samples2)).squeeze(2)
        kl += kl_knn(X=X, Y=Y, k=k).sum(dim=0)
    return kl / N


def matrix_pow(X, p):
    evals, evecs = torch.linalg.eig(X)
    return (evecs.matmul(torch.diag_embed(evals.pow(p))).matmul(evecs.transpose(1,2))).real

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


def generate_lkj(n, d,device=torch.device('cuda'), nu=1, mu=0, s=1):
    mus= torch.rand(size=(n, d))
    c = LKJCholesky(d, concentration=nu).sample((n,))
    while c.isnan().any():
        c = LKJCholesky(n, concentration=nu).sample((n,))
    #s = torch.diag_embed(LogNormal(mu,s).sample((n, d)))
    sigmas = c#torch.matmul(s, c)
    mus = mus.to(device)
    sigmas = sigmas.to(device)
    dist = MultivariateNormal(mus, scale_tril=sigmas)
    return dist

def test_kl(n, d, **kwargs):
    def generate(n, d,device=torch.device('cuda'), nu=1, mu=0, s=1):
        mus= torch.rand(size=(n, d))
        c = LKJCholesky(d, concentration=nu).sample((n,))
        #while c.isnan().any():
        #    c = LKJCholesky(n, concentration=nu).sample((n,))
        #s = torch.diag_embed(LogNormal(mu,s).sample((n, d)))
        sigmas = c#torch.matmul(s, c)
        mus = mus.to(device)
        sigmas = sigmas.to(device)
        dist = MultivariateNormal(mus, scale_tril=sigmas)
        return dist
    X, Y = generate(n, d, **kwargs), generate(n, d, **kwargs)
    t1,t2,t3,t4 = kl_nd_gaussian2(X,Y)
    return (t1+t2+t3+t4), (t1,t2,t3,t4), (X,Y)


def plot_dets(d, N=10000, nu=1.0):
    X = LKJCholesky(d, concentration=nu).sample((N,))
    Xdets=X.det()
    C = X.matmul(X.transpose(1,2))
    Cdets = C.det()
    plt.figure("Cholesky")
    plt.hist(Xdets,bins=50,density=True)
    plt.figure("Corr")
    plt.hist(Cdets,bins=50,density=True)
    plt.show()


def test_gen(d,N=100, **kwargs):
    (X,Y), (TX,TY) = PairedGaussianGenerator(return_params=True)(N, n=d)
    kl = kl_mc(TX, TY, X=X)
    return kl.mean()


def test(n, bs):
    nu,mu0,s0=1,0,1
    n_components = torch.randint(1,10,(1,)).item()
    c = LKJCholesky(n, concentration=nu).sample()
    s = torch.diag_embed(LogNormal(mu0,s0).sample((n,)))
    scale_chol = torch.matmul(s, c)
    scale = scale_chol.matmul(scale_chol.t())
    mus = MultivariateNormal(torch.zeros(n), scale_tril=scale_chol).sample((bs, n_components))
    sigmas = torch.tensor(invwishart.rvs(n+2, scale.numpy(), size=bs*n_components)).view(bs, n_components, n, n).float()
    try:
        base_dist = MultivariateNormal(mus, covariance_matrix=sigmas)
    except:
        print("Error")
    return sigmas

def test_paired(n, bs, **kwargs):
    (X,Y),(TX,TY)=PairedGaussianGenerator(return_params=True)(bs, n=n, **kwargs)
    return kl_mc(TX, TY, X=X)

'''
N=5000
bs=256
n_batches=math.ceil(N*1.0/bs)
nu=3
dims=[2,4,8,20,32]
gen=GaussianGenerator(2,return_params=True)
for dim in dims:
    kls = torch.zeros(N)
    for i in range(n_batches):
        min_j = i*bs
        max_j = min(N, (i+1)*bs)
        (X,Y),(TX,TY) = gen(max_j-min_j,n=dim,nu=nu)
        kls[min_j:max_j] = kl_mc(TX,TY,X=X)
    kl_mean = kls.mean()
    kl_sigma = (kls - kl_mean).pow(2).sum().sqrt()
    print("d=%d, mean kl=%f, stdev kl=%f"%(dim, kl_mean, kl_sigma))
'''

from scipy.special import digamma
import numpy as np

def mult_digamma(z, p):
    return digamma(z+ (1-np.arange(p))/2).sum()