from models import *
from utils import *
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class ExactDivergenceModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, latent_size=1):
        super().__init__()
        net1 = nn.Sequential(
                nn.Linear(2*input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size)
        )
        net2 = nn.Sequential(
                nn.Linear(2*input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size)
        )
        #with torch.no_grad():
         # net._modules['0'].weight.data = torch.cat([torch.eye(latent_size), torch.eye(latent_size)], dim=1)
         # net._modules['0'].weight.data.add_(torch.randn(latent_size, 2*latent_size))
        self.pair_encoder1 = RNBlock(net1, pool='max', remove_diag=True)
        self.pair_encoder2 = RNBlock(net2, pool='max')
        self.merger = nn.Sequential(
            nn.Linear(2*latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,latent_size)
        )
        #self.enc = net
        
    def forward(self, X, Y, xi=1e-5):
        N = X.size(1)
        M = Y.size(1)
        XX = torch.cat([X.unsqueeze(1).expand(-1,N,-1,-1),X.unsqueeze(2).expand(-1,-1,N,-1)],dim=-1)
        YX = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1), X.unsqueeze(2).expand(-1,-1,M,-1)], dim=-1)
        #mask = torch.eye(N, N).unsqueeze(0).unsqueeze(-1) * 99999999
        #if use_cuda:
        #    mask=mask.cuda()
        #XX = (XX.chunk(2,dim=-1)[0] - XX.chunk(2,dim=-1)[1]).norm(dim=-1, keepdim=True)
        #Z_XX = -1*torch.log(XX + mask + xi)
        #Z_YX = -1*torch.log(YX+xi)
        #Z_XX = -1*self.enc1(XX+xi) - mask
        #Z_YX = -1*self.enc2(YX+xi)
        #Z_XX = torch.max(Z_XX, dim=2)[0]
        #Z_YX = torch.max(Z_YX, dim=2)[0]
        Z_XX = self.pair_encoder1(X, X)
        Z_YX = self.pair_encoder2(X, Y)
        #Z_X = self.merger(torch.cat([Z_XX, Z_YX], dim=-1))
        Z_X = Z_YX - Z_XX
        Z_X = torch.sum(Z_X, dim=1)/N
        return -1*Z_X #+ math.log(M/(N-1))

model=ExactDivergenceModel().cuda()
losses=train(model, generate_multi(generate_gaussian_1d), simplified_divergence2, criterion=nn.MSELoss(), steps=10000, lr=1e-2, lr_decay=False)
#losses=evaluate(model7, generate_multi(generate_gaussian_1d), simplified_divergence1, criterion=nn.MSELoss(), steps=1000)

plt.plot(losses)
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Mean Squared Error")
plt.yscale("log")
plt.show()