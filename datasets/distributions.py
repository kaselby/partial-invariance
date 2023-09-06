import torch
from torch.distributions import MultivariateNormal, LKJCholesky, Categorical, MixtureSameFamily, Dirichlet, LogNormal, Bernoulli, Distribution
from scipy.stats import invwishart
import numpy as np

import os
import string

use_cuda = torch.cuda.is_available()


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
        
    
    def __call__(self, batch_size, dims=(2,6), sample_groups=1, **kwargs):
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

        #c = LKJCholesky(n, concentration=nu).sample()
        #s = torch.diag_embed(LogNormal(mu0,s0).sample((n,)))
        #scale_chol = torch.matmul(s, c)
        #scale = scale_chol.matmul(scale_chol.t())
        scale = torch.tensor(invwishart.rvs(n+2, torch.eye(n).numpy())).float()

        def _generate_set():
            mus = MultivariateNormal(torch.zeros(n), covariance_matrix=scale).sample((batch_size, n_components))
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
    def __init__(self, return_params=False, variable_dim=False, max_rho=0.999):
        self.return_params=return_params
        self.variable_dim=variable_dim
        self.max_rho=max_rho
        
    def _build_dist(self, batch_size, corr, n):
        mu = torch.zeros((batch_size, n*2))
        I = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
        if use_cuda:
            I = I.cuda()
            mu = mu.cuda()
        rhoI = corr.unsqueeze(-1).unsqueeze(-1) * I
        cov = torch.cat([torch.cat([I, rhoI], dim=1), torch.cat([rhoI, I], dim=1)], dim=2)
        return MultivariateNormal(mu, covariance_matrix=cov)

    def _generate(self, batch_size, n, set_size=(100,150), sample_groups=1, corr=None):
        n_samples = torch.randint(*set_size,(1,))
        if corr is None:
            corr = self.max_rho-2*self.max_rho*(torch.rand((batch_size,)))
            if use_cuda:
                corr = corr.cuda()
        dists = self._build_dist(batch_size, corr, n)
        X, Y = dists.sample(n_samples*sample_groups).transpose(0,1).chunk(2, dim=-1)

        if self.return_params:
            return (X, Y), (corr,)
        else:
            return X, Y

    def __call__(self, batch_size, dims=(2,6), sample_groups=1, **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        return self._generate(batch_size, sample_groups=sample_groups, **kwargs)

class StandardGaussianGenerator():
    def __call__(self, batch_size, n, sample_groups=1, set_size=(100,150), n_samples=-1):
        if n_samples <= 0:
            n_samples = torch.randint(*set_size,(1,))
        X = torch.randn(batch_size, n_samples*sample_groups, n)
        return X


class CorrelatedGaussianGenerator2():
    def __init__(self, return_params=False, variable_dim=False, max_rho=0.999):
        self.return_params=return_params
        self.variable_dim=variable_dim
        self.max_rho=max_rho
        
    def _build_dist(self, batch_size, corr, n):
        mu = torch.zeros((batch_size, 2*n))
        I = torch.eye(n).unsqueeze(0).expand(batch_size, -1, -1)
        if use_cuda:
            I = I.cuda()
            mu = mu.cuda()
        rhoI = corr.unsqueeze(-1).unsqueeze(-1) * I
        cov = torch.cat([torch.cat([I, rhoI], dim=1), torch.cat([rhoI, I], dim=1)], dim=2)
        return MultivariateNormal(mu, covariance_matrix=cov)

    def _generate(self, batch_size, n, set_size=(100,150), sample_groups=1, corr=None):
        n_samples = torch.randint(*set_size,(1,))
        if corr is None:
            corr = self.max_rho-2*self.max_rho*(torch.rand((batch_size,)))
            if use_cuda:
                corr = corr.cuda()
        joint_dist = self._build_dist(batch_size, corr, n)
        P = joint_dist.sample(n_samples*sample_groups).transpose(0,1)
        marginal = MultivariateNormal(torch.zeros(n*2).cuda(), covariance_matrix=torch.eye(n*2).cuda())
        Q = marginal.sample(batch_size * n_samples * sample_groups).view(batch_size, -1, n*2)

        if self.return_params:
            return (P, Q), (joint_dist,marginal)
        else:
            return P,Q

    def __call__(self, batch_size, dims=(2,6), sample_groups=1, **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item() * 2
            kwargs['n'] = n
        return self._generate(batch_size, sample_groups=sample_groups, **kwargs)

from datasets.flows import MADE, MADE_IAF, BatchNormFlow, Reverse, FlowSequential, BatchOfFlows
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

    def __call__(self, batch_size, dims=(2,6), sample_groups=1, **kwargs):
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
        


class KroneckerProduct(Distribution):
    def __init__(self, *distributions):
        super().__init__()
        self.distributions = distributions

    def sample(self, sample_shape=torch.Size([])):
        samples = [dst.sample(sample_shape) for dst in self.distributions]
        return samples
    
    def log_prob(self, value):
        logprobs = [dst.log_prob(x) for x,dst in zip(value, self.distributions)]
        return sum(logprobs)
        
class Mixture(Distribution):
    def __init__(self, mixing_distribution, base_distribution, mixture_dim=1):
        super().__init__()
        self.base_distribution = base_distribution
        self.mixing_distribution = mixing_distribution
        self.mixture_dim=mixture_dim
        self._event_shape = self.base_distribution.event_shape
        self._event_ndims = len(self.event_shape)

    def sample(self, sample_shape=torch.Size([])):
        mixture_labels = self.mixing_distribution.sample(sample_shape).unsqueeze(-1)
        mixture_labels = mixture_labels.view(*mixture_labels.size(), *[1 for _ in range(self._event_ndims)])
        mixture_labels = mixture_labels.expand(*mixture_labels.size()[:-1], self.event_shape[-1])
        all_samples = self.base_distribution.sample(sample_shape)
        mixing_dim = self.mixture_dim + len(sample_shape)
        outputs = torch.gather(all_samples, mixing_dim, mixture_labels).squeeze(mixing_dim)
        return outputs, mixture_labels
    
    def log_prob(self, x, labels=None):
        x = self._pad(x)
        log_prob_x = self.base_distribution.log_prob(x)  # [S, B, k]

        if labels is not None:
            label_logprobs = self.mixing_distribution.log_prob(labels)
            component_logprobs = torch.gather(log_prob_x, -1, labels.view(1,1,-1)).squeeze(-1)
            return label_logprobs + component_logprobs
        else:
            log_mix_prob = torch.log_softmax(self.mixing_distribution.logits,
                                    dim=-1)  # [B, k]
            return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)





class LabelledGaussianGenerator():
    def __init__(self, return_params=False, variable_dim=False):
        self.return_params=return_params
        self.variable_dim=variable_dim

    def _generate(self, batch_size, n, return_params=False, set_size=(100,150), sample_groups=1):
        mus = torch.rand(batch_size, 2, n)
        sigmas = torch.rand(batch_size, 2, n)
        sigmas = torch.diag_embed(sigmas)
        dist = MultivariateNormal(mus, covariance_matrix=sigmas)
        mixing_dist = Categorical(torch.ones(batch_size, 2))

        joint = Mixture(mixing_dist, dist)
        marginal = KroneckerProduct(MixtureSameFamily(mixing_dist, dist))

        n_samples = torch.randint(*set_size,(1,))
        X, labels = joint.sample(n_samples * sample_groups)
        X = X.transpose(0,1)
        labels = labels.transpose(0,1)

        if self.return_params:
            return (X, labels), (joint, marginal)
        else:
            return X, labels

    def __call__(self, batch_size, dims=(2,6), sample_groups=1, **kwargs):
        if self.variable_dim:
            n = torch.randint(*dims,(1,)).item()
            kwargs['n'] = n
        return self._generate(batch_size, sample_groups=sample_groups, **kwargs)



