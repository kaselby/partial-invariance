from models import *
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def nn_dist(X):
  dists = knn(X, 1)
  return dists

class MaskedESAB(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, ln=False):
        super(MaskedESAB, self).__init__()
        self.mab = EquiMAB3(input_size, latent_size, num_heads, ln=ln)
        #self.mab = EquiMAB2(latent_size, num_heads, ln=ln)

    def forward(self, X):
      mask = (1 - torch.eye(X.size(1))).cuda()
      return self.mab(X, X, mask=mask)

class EquiEncoder(nn.Module):
  def __init__(self, latent_size, n_blocks, n_heads, ln=False):
    super().__init__()
    self.enc = nn.Sequential(
                MaskedESAB(1, latent_size, n_heads, ln=ln),
                *[MaskedESAB(latent_size, latent_size, n_heads, ln=ln) for i in range(n_blocks-1)],
    )
    self.out = nn.Linear(latent_size, 1)

  def forward(self, X):
    Z = self.enc(X.unsqueeze(-1)).max(dim=2)[0]
    return self.out(Z)

class EquiMultiSetTransformer1(nn.Module):
    def __init__(self, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, num_blocks=2, ln=False, remove_diag=False):
        super().__init__()
        self.proj = nn.Linear(1, dim_hidden)
        self.enc = nn.Sequential(*[EquiCSAB(dim_hidden, dim_hidden, num_heads, ln=ln, remove_diag=remove_diag) for i in range(num_blocks)])
        self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(2*dim_hidden, dim_output),)

    def forward(self, X, Y):
        ZX, ZY = self.enc((self.proj(X.unsqueeze(-1)),self.proj(Y.unsqueeze(-1))))
        ZX = ZX.sum(dim=2)
        ZY = ZY.sum(dim=2)
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)

model=EquiMultiSetTransformer1(1,1, dim_hidden=32, ln=True, remove_diag=True, num_blocks=2).cuda()
losses=train(model, generate_multi(generate_gaussian_nd), wasserstein, exact_loss=False, criterion=nn.MSELoss(), steps=15000, lr=1e-3, n=2, set_size=(50,75))
      
'''
d=2
hs=32
nh=4
ln=True
n_blocks=2
model= EquiEncoder(hs, n_blocks, nh, ln).cuda()
losses=train(model, generate_gaussian_nd, wasserstein, criterion=nn.MSELoss(), steps=20000, lr=1e-3, n=2, set_size=(50,75))
'''

print(sum(losses[-50:])/50)
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=2, set_size=(5,6))
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=3, set_size=(5,6))
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=4, set_size=(5,6))