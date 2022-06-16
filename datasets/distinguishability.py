import torch
from torch.distributions import MultivariateNormal, LKJCholesky, Categorical, MixtureSameFamily, Dirichlet, LogNormal

class DistinguishabilityGenerator():
    def __init__(self, device=torch.device('cpu')):
        self.device=device
    

    def _generate_gmm(self, batch_size, n, p=0.5, set_size=(100,150), component_range=(1,5), nu=5, mu0=0, s0=0.3):
        def _generate_mixture(batch_size, n, component_range, nu, mu0, s0):
            n_components = torch.randint(*component_range,(1,)).item()
            mus= torch.rand(size=(batch_size, n_components, n))
            c = LKJCholesky(n, concentration=nu).sample((batch_size, n_components))
            while c.isnan().any():
                c = LKJCholesky(n, concentration=nu).sample((batch_size, n_components))
            s = torch.diag_embed(LogNormal(mu0,s0).sample((batch_size, n_components, n)))
            sigmas = torch.matmul(s, c)
            mus = mus.to(self.device)
            sigmas = sigmas.to(self.device)
            logits = Dirichlet(torch.ones(n_components).to(self.device)/n_components).sample((batch_size,))
            base_dist = MultivariateNormal(mus, scale_tril=sigmas)
            mixing_dist = Categorical(logits=logits)
            dist = MixtureSameFamily(mixing_dist, base_dist)
            return dist
        n_samples = torch.randint(*set_size,(1,))
        aligned = (torch.rand(batch_size) < p).to(self.device)
        X_dists = _generate_mixture(batch_size, n, component_range, nu, mu0, s0)
        Y_dists = _generate_mixture(batch_size, n, component_range, nu, mu0, s0)

        X = X_dists.sample(n_samples).transpose(0,1).float()
        Y_aligned = X_dists.sample(n_samples).transpose(0,1).float()
        Y_unaligned = Y_dists.sample(n_samples).transpose(0,1).float()
        Y = torch.where(aligned.view(-1, 1, 1), Y_aligned, Y_unaligned)

        return (X, Y), aligned.float()

    def __call__(self, *args, **kwargs):
        return self._generate_gmm(*args, **kwargs)





class SetDataGenerator():
    def __init__(self, dataset, device=torch.device('cpu')):
        self.dataset = dataset
        self.device=device
    
    def _generate(self, batch_size, set_size=(25,50), n_samples=-1):
        indices = torch.randperm(len(self.dataset))
        n_samples = torch.randint(*set_size, (1,)).item() if n_samples <= 0 else n_samples

        batch = []
        for i in range(batch_size):
            set_i = [self.dataset[j.item()] for j in indices[i*n_samples:(i+1)*n_samples]]
            batch.append(torch.stack(set_i, 0))
        batch = torch.stack(batch, 0)

        return batch.to(self.device)
    
    def __call__(self, *args, **kwargs):
        return self._generate(*args, **kwargs)