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
      

d=2
hs=32
nh=4
ln=True
n_blocks=2
model= EquiEncoder(hs, n_blocks, nh, ln).cuda()
losses=train(model, generate_gaussian_nd, wasserstein, criterion=nn.MSELoss(), steps=20000, lr=1e-3, n=2, set_size=(50,75))

print(sum(losses[-50:])/50)
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=2, set_size=(5,6))
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=3, set_size=(5,6))
show_examples(model, generate_gaussian_nd, wasserstein, samples=4, n=4, set_size=(5,6))
