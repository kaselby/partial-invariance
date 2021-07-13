import torch.nn as nn
import torch

use_cuda = torch.cuda.is_available()

class SetModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, X):
        return self.decoder(self.encoder(X).sum(dim=1))

class MultiSetModel(nn.Module):
    def __init__(self, encoder1, encoder2, decoder):
        super().__init__()
        self.encoder1=encoder1
        self.encoder2=encoder2
        self.decoder=decoder

    def forward(self, X, Y):
        Z_x = self.encoder1(X).sum(dim=1)
        Z_y = self.encoder1(Y).sum(dim=1)
        return self.decoder(torch.cat([Z_x,Z_y], dim=-1))

class RelationNetwork(nn.Module):
    def __init__(self, encoder, decoder, pooling='sum'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pooling = pooling
    
    def forward(self, X):
        pairs = torch.cat([X.unsqueeze(1), X.unsqueeze(2)], dim=-1).view(X.size(0),-1, X.size(-1)*2)
        Z = self.encoder(pairs)
        if self.pooling == 'sum':
            Z = Z.sum(dim=1)
        elif self.pooling == 'max':
            Z = Z.max(dim=1)
        else:
            raise NotImplementedError()
        return self.decoder(Z)

class RNBlock(nn.Module):
    def __init__(self, net, remove_diag=False, pool='sum') -> None:
        super().__init__()
        self.net = net
        self.remove_diag = remove_diag
        self.pool = pool
    
    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1), X.unsqueeze(2).expand(-1,-1,M,-1)], dim=-1)
        Z = self.net(pairs)
        if self.remove_diag:
            mask = torch.eye(N, N).unsqueeze(0).unsqueeze(-1) * -999999999
            if use_cuda:
                mask=mask.cuda()
            Z = Z + mask
        if self.pool == 'sum':
            Z = torch.sum(Z, dim=2)
        elif self.pool == 'max':
            Z = torch.max(Z, dim=2)[0]
        else:
            raise NotImplementedError()
        return Z

class RNModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        N = X.size(1)
        Z = self.encoder(X, X)
        Z = torch.sum(Z, dim=1)
        return self.decoder(Z)

class MultiRNModel(nn.Module):
    def __init__(self, encoder1, encoder2, merger, decoder):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.merger = merger
        self.decoder = decoder

    def forward(self, X, Y):
        Z_XX = self.encoder1(X, Y)
        Z_YX = self.encoder2(X, Y)
        Z = self.merger(torch.cat([Z_XX, Z_YX],dim=-1))
        Z = torch.mean(Z, dim=1)
        return self.decoder(Z)


import copy
class DivergenceRN(nn.Module):
    def __init__(self, pair_encoder, merge_encoder, decoder):
        super().__init__()
        self.e1_xx = pair_encoder
        self.e1_xy = copy.deepcopy(pair_encoder)
        self.e1_yx = copy.deepcopy(pair_encoder)
        self.e1_yy = copy.deepcopy(pair_encoder)
        self.e2_x = merge_encoder
        self.e2_y = copy.deepcopy(merge_encoder)
        self.decoder = decoder

    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        XX = torch.cat([X.unsqueeze(1).expand(-1,N,-1,-1), X.unsqueeze(2).expand(-1,-1,N,-1)], dim=-1)
        YY = torch.cat([Y.unsqueeze(1).expand(-1,M,-1,-1), Y.unsqueeze(2).expand(-1,-1,M,-1)], dim=-1)
        XY = torch.cat([X.unsqueeze(1).expand(-1,M,-1,-1), Y.unsqueeze(2).expand(-1,-1,N,-1)], dim=-1)
        YX = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1), X.unsqueeze(2).expand(-1,-1,M,-1)], dim=-1)
        Z_XX = self.e1_xx(XX)
        Z_YY = self.e1_yy(YY)
        Z_XY = self.e1_xy(XY)
        Z_YX = self.e1_yx(YX)
        Z_XX = torch.max(Z_XX, dim=2)[0]
        Z_YY = torch.max(Z_YY, dim=2)[0]
        Z_XY = torch.max(Z_XY, dim=2)[0]
        Z_YX = torch.max(Z_YX, dim=2)[0]
        #Z_X = Z_XX/Z_YX
        #Z_Y = Z_YY/Z_XY
        Z_X = Z_XX - Z_YX
        Z_Y = Z_XX - Z_YX
        #Z_X = self.e2_x(torch.cat([Z_XX, Z_YX],dim=-1))
        #Z_Y = self.e2_y(torch.cat([Z_YY, Z_XY],dim=-1))
        Z_X = torch.sum(Z_X, dim=1)
        Z_Y = torch.sum(Z_Y, dim=1)
        return self.decoder(torch.cat([Z_X, Z_Y], dim=-1))

class ExactDivergenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        XX = (X.unsqueeze(1).expand(-1,N,-1,-1) - X.unsqueeze(2).expand(-1,-1,N,-1)).norm(dim=-1)
        YX = (Y.unsqueeze(1).expand(-1,N,-1,-1) - X.unsqueeze(2).expand(-1,-1,M,-1)).norm(dim=-1)
        mask = torch.eye(N, N).unsqueeze(0)* 999999999
        if use_cuda:
            mask=mask.cuda()
        Z_XX = -1*torch.log(XX + mask)
        Z_YX = -1*torch.log(YX)
        Z_XX = torch.max(Z_XX, dim=2)[0]
        Z_YX = torch.max(Z_YX, dim=2)[0]
        Z_X = Z_XX - Z_YX
        Z_X = torch.sum(Z_X, dim=1)/N
        return -1*Z_X



class EquiNN(nn.Module):
    def __init__(self):
        self.l = nn.Parameter(torch.empty(1))
        self.g = nn.Parameter(torch.empty(1))

    def forward(self, X, maxpool=False):
        N = X.size(-1)
        if maxpool:
            return X.matmul(self.l * torch.eye(N)) + self.g * X.max(dim=-1)[0] * torch.ones_like(X)
        else:
            return X.matmul(self.l * torch.eye(N) + self.g*torch.ones(N, N))
        

class PEquiNN(nn.Module):
    def __init__(self):
        self.l_xx = nn.Parameter(torch.empty(1))
        self.l_yy = nn.Parameter(torch.empty(1))
        self.g_xx = nn.Parameter(torch.empty(1))
        self.g_xy = nn.Parameter(torch.empty(1))
        self.g_yx = nn.Parameter(torch.empty(1))
        self.g_yy = nn.Parameter(torch.empty(1))

    def forward(self, X, Y):
        n = X.size(-1)
        m = Y.size(-1)
        theta_xx = torch.eye(n) * self.l_xx + torch.ones(n,n) * self.g_xx
        theta_yy = torch.eye(m) * self.l_yy + torch.ones(m,m) * self.g_yy
        theta_xy = torch.ones(m, n) * self.g_xy
        theta_yx = torch.ones(n, m) * self.g_yx
        return X.mm(theta_xx) + Y.mm(theta_yx), Y.mm(theta_yy) + X.mm(theta_xy)


def simple_model(input_size, output_size, latent_size=16, hidden_size=32):
    encoder = nn.Linear(input_size, latent_size, bias=False)
    nn.init.eye_(encoder.weight)
    decoder = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    return SetModel(encoder, decoder)


def simple_multi_model(input_size, output_size, latent_size=16, hidden_size=32):
    encoder1 = nn.Linear(input_size, latent_size, bias=False)
    encoder2 = nn.Linear(input_size, latent_size, bias=False)
    nn.init.eye_(encoder1.weight)
    nn.init.eye_(encoder2.weight)
    decoder = nn.Sequential(
        nn.Linear(2*latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    return MultiSetModel(encoder1, encoder2, decoder)


def entropy_model(input_size, output_size, latent_size=4, hidden_size=12, num_blocks=1):
    blocks=[]
    for i in range(num_blocks):
        if i == 0:
            blocks.append(RNBlock(nn.Sequential(
                nn.Linear(2*input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size),
            )))
        else:
            blocks.append(RNBlock(nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_size),
            )))
    
    encoder = nn.Sequential(*blocks)
    decoder = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    return MultiSetModel(encoder, decoder)

def divergence_model(input_size, output_size, latent_size=4, hidden_size=16):
    pair_encoder = nn.Sequential(
        nn.Linear(2*input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, latent_size),
    )
    merge_encoder = nn.Sequential(
        nn.Linear(2*latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, latent_size),
    )
    decoder = nn.Sequential(
        nn.Linear(2*latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    for module in decoder.modules():
        if type(module) == nn.Linear:
            nn.init.eye_(module.weight)
    return DivergenceRN(pair_encoder, merge_encoder, decoder)