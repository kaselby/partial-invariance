import torch.nn as nn
import torch
import torch.nn.functional as F
import math

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

class MultiRNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, remove_diag=False, pool='sum') -> None:
        super().__init__()
        self.e_xx = RNBlock(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size)), pool=pool, remove_diag=remove_diag)
        self.e_xy = RNBlock(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size)), pool=pool, remove_diag=False)
        self.e_yx = RNBlock(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size)), pool=pool, remove_diag=False)
        self.e_yy = RNBlock(nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size)), pool=pool, remove_diag=remove_diag)
        self.fc_X = nn.Linear(2*latent_size, latent_size)
        self.fc_Y = nn.Linear(2*latent_size, latent_size)
        self.remove_diag = remove_diag
        self.pool = pool
    
    def forward(self, X, Y):
        Z_XX = self.e_xx(X, X)
        Z_XY = self.e_xy(X, Y)
        Z_YX = self.e_yx(Y, X)
        Z_YY = self.e_yy(Y, Y)
        X_out = F.relu(self.fc_X(torch.cat([Z_XX, Z_XY], dim=-1)))
        Y_out = F.relu(self.fc_Y(torch.cat([Z_YY, Z_YX], dim=-1)))

        return X_out, Y_out

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
    def __init__(self, input_size, output_size, hidden_size, latent_size, **kwargs):
        super().__init__()
        self.encoder = MultiRNBlock(input_size, hidden_size, latent_size, **kwargs)
        self.decoder = nn.Sequential(nn.Linear(2*latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))

    def forward(self, X, Y):
        Z_X, Z_Y = self.encoder(X, Y)
        Z_X = torch.max(Z_X, dim=1)[0]
        Z_Y = torch.max(Z_Y, dim=1)[0]
        out = self.decoder(torch.cat([Z_X, Z_Y], dim=-1))
        return out


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
    def __init__(self, bias=False, maxpool=False):
        super().__init__()
        self.maxpool = maxpool
        self.bias = bias

        self.l = nn.Parameter(torch.empty(1))
        self.g = nn.Parameter(torch.empty(1))
        if self.bias:
            self.b = nn.Parameter(torch.empty(1))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.l)
        nn.init.uniform_(self.g)
        if self.bias:
            nn.init.uniform_(self.b)

    def forward(self, X):
        N = X.size(-1)
        if self.maxpool:
            out= X.matmul(self.l * torch.eye(N, device=X.device)) + self.g * X.max(dim=-1)[0] * torch.ones_like(X, device=X.device)
        else:
            out= X.matmul(self.l * torch.eye(N, device=X.device) + self.g*torch.ones(N, N, device=X.device)) 

        if self.bias:
            out += self.b * torch.ones_like(X, device=X.device)
        
        return out


class EquiLinearLayer1(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Lambda = nn.Linear(input_size, output_size, bias=False)
        self.Gamma = nn.Linear(input_size, output_size, bias=False)

    def forward(self, X):
        m = X.size(-2)
        return self.Lambda(X) - self.Gamma(torch.ones(m, m, device=X.device).matmul(X))

class EquiLinearLayer2(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Lambda = nn.Linear(input_size, output_size, bias=False)
        self.Gamma = nn.Linear(input_size, output_size, bias=False)

    def forward(self, X):
        m = X.size(-2)
        return self.Lambda(X) - self.Gamma(torch.ones(m,1, device=X.device).matmul(X.max(dim=-2, keepdim=True)[0]))

class EquiLinearLayer3(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Gamma = nn.Linear(input_size, output_size, bias=True)

    def forward(self, X):
        m = X.size(-2)
        return self.Gamma(X - torch.ones(m,1, device=X.device).matmul(X.max(dim=-2,keepdim=True)[0]))

        
class EquiLinearBlock1(nn.Module):
    def __init__(self, hidden_size, num_layers):
        assert num_layers >= 1
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            *[x for i in range(num_layers-1) for x in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]],
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

class EquiLinearBlock2(nn.Module):
    def __init__(self, hidden_size, num_layers, layer=EquiLinearLayer1):
        assert num_layers >= 1
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.net = nn.Sequential(
            layer(1, hidden_size),
            nn.ReLU(),
            *[x for i in range(num_layers-1) for x in [layer(hidden_size, hidden_size), nn.ReLU()]],
            layer(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)

class PEquiNN(nn.Module):
    def __init__(self, bias=False, f=nn.ReLU):
        super().__init__()
        self.bias=bias

        self.l_xx = nn.Parameter(torch.empty(1))
        self.l_yy = nn.Parameter(torch.empty(1))
        self.g_xx = nn.Parameter(torch.empty(1))
        self.g_xy = nn.Parameter(torch.empty(1))
        self.g_yx = nn.Parameter(torch.empty(1))
        self.g_yy = nn.Parameter(torch.empty(1))
        if self.bias:
            self.b_x = nn.Parameter(torch.empty(1))
            self.b_y = nn.Parameter(torch.empty(1))

    def _init_weights(self):
        nn.init.uniform_(self.l_xx)
        nn.init.uniform_(self.l_yy)
        nn.init.uniform_(self.g_xx)
        nn.init.uniform_(self.g_xy)
        nn.init.uniform_(self.g_yx)
        nn.init.uniform_(self.g_yy)
        if self.bias:
            nn.init.uniform_(self.b_x)
            nn.init.uniform_(self.b_y)

    def forward(self, X, Y):
        n = X.size(-1)
        m = Y.size(-1)
        theta_xx = torch.eye(n, device=X.device) * self.l_xx + torch.ones(n,n, device=X.device) * self.g_xx
        theta_yy = torch.eye(m, device=X.device) * self.l_yy + torch.ones(m,m, device=X.device) * self.g_yy
        theta_xy = torch.ones(m, n, device=X.device) * self.g_xy
        theta_yx = torch.ones(n, m, device=X.device) * self.g_yx
        o_x = X.mm(theta_xx) + Y.mm(theta_yx)
        o_y = Y.mm(theta_yy) + X.mm(theta_xy)
        if self.bias:
            o_x += self.b_x
            o_y += self.b_y
        return o_x, o_y

class PEquiLinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.Lambda_xx = nn.Linear(input_size, output_size, bias=False)
        self.Lambda_yy = nn.Linear(input_size, output_size, bias=False)
        self.Gamma_xx = nn.Linear(input_size, output_size, bias=False)
        self.Gamma_xy = nn.Linear(input_size, output_size, bias=False)
        self.Gamma_yx = nn.Linear(input_size, output_size, bias=False)
        self.Gamma_yy = nn.Linear(input_size, output_size, bias=False)

    def forward(self, X, Y):
        m_x = X.size(-2)
        m_y = Y.size(-2)
        out_xx = self.Lambda_xx(X) + self.Gamma_xx(torch.ones(m_x,1, device=X.device).matmul(X.max(dim=-2, keepdim=True)[0]))
        out_yy = self.Lambda_yy(Y) + self.Gamma_yy(torch.ones(m_y,1, device=X.device).matmul(Y.max(dim=-2, keepdim=True)[0]))
        out_xy = self.Gamma_xy(torch.ones(m_y,1, device=X.device).matmul(X.max(dim=-2, keepdim=True)[0]))
        out_yx = self.Gamma_yx(torch.ones(m_x,1, device=X.device).matmul(Y.max(dim=-2, keepdim=True)[0]))

        out_x = out_xx + out_yx
        out_y = out_yy + out_xy

        return out_x, out_y

class PEquiLinearBlock1(nn.Module):
    def __init__(self, hidden_size, num_layers, layer=PEquiLinearLayer):
        assert num_layers >= 1
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.net = nn.Sequential(
            layer(1, hidden_size),
            nn.ReLU(),
            *[x for i in range(num_layers-1) for x in [layer(hidden_size, hidden_size), nn.ReLU()]],
            layer(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x.unsqueeze(-1)).squeeze(-1)


class EquiEncoder(nn.Module):
    def __init__(self, latent_size, input_size=1, set_dim=-2, pool='max'):
        super().__init__()
        self.input_size=input_size
        self.set_dim=set_dim
        self.pool=pool
        self.net = nn.Linear(input_size, latent_size)

    def forward(self, X):
        if self.input_size == 1:
            Z = self.net(X.unsqueeze(-1))
        else:
            Z = self.net(X)

        if self.pool == 'sum':
            Z = torch.sum(Z, dim=self.set_dim)
        elif self.pool == 'max':
            Z = torch.max(Z, dim=self.set_dim)[0]
        else:
            raise NotImplementedError()
        return Z


class EquiRNBlock1(nn.Module):
    def __init__(self, latent_size, hidden_size, layer=EquiLinearLayer2, equi_layers=1, pool='max', remove_diag=False):
        super().__init__()
        self.pool = pool
        self.remove_diag=remove_diag
        self.eq = nn.Sequential(*[
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ])
        self.enc = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
        )
    
    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1).unsqueeze(-1), X.unsqueeze(2).expand(-1,-1,M,-1).unsqueeze(-1)], dim=-1)
        Z = self.eq(pairs).squeeze(-1)
        Z = Z.sum(dim=-1, keepdim=True)
        Z = self.enc(Z)
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

class EquiRNBlock2(nn.Module):
    def __init__(self, latent_size, hidden_size, layer=EquiLinearLayer2, equi_layers=1, pool='max', remove_diag=False):
        super().__init__()
        self.pool = pool
        self.remove_diag=remove_diag
        self.eq = nn.Sequential(*[
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        ])
    
    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1).unsqueeze(-1), X.unsqueeze(2).expand(-1,-1,M,-1).unsqueeze(-1)], dim=-1)
        Z = self.eq(pairs)
        Z = Z.sum(dim=3, keepdim=False)#[0]
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


#
#   From Set Transformers
#
def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, **kwargs)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        E = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=2)
        else:
            A = torch.softmax(E, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class EquiMAB1(nn.Module):
    def __init__(self, num_heads, ln=False):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(1, num_heads)
        self.fc_k = nn.Linear(1, num_heads)
        self.fc_v = nn.Linear(1, num_heads)
        self.fc_o = nn.Linear(num_heads, 1)
    
    def forward(self, Q, K):
        inp_size = Q.size(-1)
        Q = self.fc_q(Q.unsqueeze(-1))
        K, V = self.fc_k(K.unsqueeze(-1)), self.fc_v(K.unsqueeze(-1))

        Q_ = Q.permute(3,0,1,2)#torch.cat(Q.split(self.num_heads, 3), 0)
        K_ = K.permute(3,0,1,2)#torch.cat(K.split(self.num_heads, 3), 0)
        V_ = V.permute(3,0,1,2)#torch.cat(V.split(self.num_heads, 3), 0)

        E = Q_.matmul(K_.transpose(2,3))
        A = torch.softmax(E, 2)
        O = (Q_ + A.matmul(V_)).permute(1,2,3,0)
        #O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = F.relu(self.fc_o(O)).squeeze(-1)
        #O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class EquiMAB2(nn.Module):
    def __init__(self, latent_size, num_heads, ln=False):
        super().__init__()
        self.latent_size=latent_size
        self.num_heads = num_heads
        self.fc_q = nn.Sequential(nn.Linear(1, latent_size), nn.ReLU(), nn.Linear(latent_size, latent_size))
        self.fc_k = nn.Sequential(nn.Linear(1, latent_size), nn.ReLU(), nn.Linear(latent_size, latent_size))
        self.fc_v = nn.Sequential(nn.Linear(1, latent_size), nn.ReLU(), nn.Linear(latent_size, latent_size))
        self.fc_o = nn.Linear(latent_size, latent_size)
    
    def forward(self, Q, K, mask=None):
        inp_size = Q.size(-1)
        Q = self.fc_q(Q.unsqueeze(-1))
        K, V = self.fc_k(K.unsqueeze(-1)), self.fc_v(K.unsqueeze(-1))

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 3), 0)
        K_ = torch.cat(K.split(dim_split, 3), 0)
        V_ = torch.cat(V.split(dim_split, 3), 0)

        Q_ = torch.max(Q_, dim=2)[0]
        K_ = torch.max(K_, dim=2)[0]
        V_ = torch.max(V_, dim=2)[0]

        E = Q_.matmul(K_.transpose(1,2)/math.sqrt(self.latent_size))
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=2)
        else:
            A = torch.softmax(E, 2)
        O = torch.cat((Q_ + A.matmul(V_)).split(Q.size(0), 0), 2)
        
        #O = torch.max(O, dim=2)[0]

        #O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        #O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O



class EquiMAB3(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, ln=False):
        super().__init__()
        self.latent_size=latent_size
        self.num_heads = num_heads
        self.fc_q = nn.Linear(input_size, latent_size)
        self.fc_k = nn.Linear(input_size, latent_size)
        self.fc_v = nn.Linear(input_size, latent_size)
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)
        self.fc_o = nn.Linear(latent_size, latent_size)
    
    def forward(self, Q, K, mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 3), 0)
        K_ = torch.cat(K.split(dim_split, 3), 0)
        V_ = torch.cat(V.split(dim_split, 3), 0)

        E = Q_.transpose(1,2).matmul(K_.transpose(1,2).transpose(2,3)).sum(dim=1) / math.sqrt(self.latent_size)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=2)
        else:
            A = torch.softmax(E, 2)
        O = Q_ + A.matmul(V_.view(*V_.size()[:-2], -1)).view(*Q_.size())
        O = torch.cat(O.split(Q.size(0), 0), 3)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O





class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class EquiSAB1(nn.Module):
    def __init__(self, num_heads, ln=False):
        super().__init__()
        self.mab = EquiMAB1(num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class EquiSAB2(nn.Module):
    def __init__(self, latent_size, num_heads, ln=False):
        super().__init__()
        self.mab = EquiMAB2(latent_size, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class EquiSAB3(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, ln=False):
        super().__init__()
        self.mab = EquiMAB3(input_size, latent_size, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class CSAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(CSAB, self).__init__()
        self.SAB_X = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.SAB_Y = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.SAB_XY = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.SAB_YX = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.fc_X = nn.Linear(dim_out*2, dim_out)
        self.fc_Y = nn.Linear(dim_out*2, dim_out)

    def forward(self, inputs):
        X, Y = inputs
        XX = self.SAB_X(X, X)
        YY = self.SAB_Y(Y, Y)
        XY = self.SAB_XY(X, Y)
        YX = self.SAB_YX(Y, X)
        X_out = X + F.relu(self.fc_X(torch.cat([XX, XY], dim=-1)))
        Y_out = Y + F.relu(self.fc_Y(torch.cat([YY, YX], dim=-1)))
        return (X_out, Y_out)

class EquiCSAB(nn.Module):
    def __init__(self, input_size, latent_size, num_heads, ln=False, remove_diag=False):
        super().__init__()
        self.SAB_X = EquiMAB3(input_size, latent_size, num_heads, ln=ln)
        self.SAB_Y = EquiMAB3(input_size, latent_size, num_heads, ln=ln)
        self.SAB_XY = EquiMAB3(input_size, latent_size, num_heads, ln=ln)
        self.SAB_YX = EquiMAB3(input_size, latent_size, num_heads, ln=ln)
        self.fc_X = nn.Linear(2*latent_size, latent_size)
        self.fc_Y = nn.Linear(2*latent_size, latent_size)
        self.remove_diag=remove_diag

    def forward(self, inputs):
        X, Y = inputs
        if self.remove_diag:
            mask_x = (1 - torch.eye(X.size(1))).cuda()
            mask_y = (1 - torch.eye(Y.size(1))).cuda()
        else:
            mask_x, mask_y=None,None
        XX = self.SAB_X(X, X, mask=mask_x)
        YY = self.SAB_Y(Y, Y, mask=mask_y)
        XY = self.SAB_XY(X, Y)
        YX = self.SAB_YX(Y, X)
        X_out = X + F.relu(self.fc_X(torch.cat([XX, XY], dim=-1)))
        Y_out = Y + F.relu(self.fc_Y(torch.cat([YY, YX], dim=-1)))
        return (X_out, Y_out)

class CSAB2(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(CSAB2, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q_x = nn.Linear(dim_Q, dim_V)
        self.fc_k_x = nn.Linear(dim_K, dim_V)
        self.fc_v_x = nn.Linear(dim_K, dim_V)
        self.fc_q_y = nn.Linear(dim_Q, dim_V)
        self.fc_k_y = nn.Linear(dim_K, dim_V)
        self.fc_v_y = nn.Linear(dim_K, dim_V)
        self.fc_X = nn.Linear(dim_V*2, dim_V)
        self.fc_Y = nn.Linear(dim_V*2, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, inputs):
        X, Y = inputs
        Q_x = self.fc_q_x(X)
        K_x, V_x = self.fc_k_x(X), self.fc_v_x(X)
        Q_y = self.fc_q_y(Y)
        K_y, V_y = self.fc_k_y(Y), self.fc_v_y(Y)

        dim_split = self.dim_V // self.num_heads
        Q_x_ = torch.cat(Q_x.split(dim_split, 2), 0)
        K_x_ = torch.cat(K_x.split(dim_split, 2), 0)
        V_x_ = torch.cat(V_x.split(dim_split, 2), 0)
        Q_y_ = torch.cat(Q_y.split(dim_split, 2), 0)
        K_y_ = torch.cat(K_y.split(dim_split, 2), 0)
        V_y_ = torch.cat(V_y.split(dim_split, 2), 0)

        A_xx = torch.softmax(Q_x_.bmm(K_x_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A_xy = torch.softmax(Q_x_.bmm(K_y_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A_yx = torch.softmax(Q_y_.bmm(K_x_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A_yy = torch.softmax(Q_y_.bmm(K_y_.transpose(1,2))/math.sqrt(self.dim_V), 2)

        O_xx = torch.cat((Q_x_ + A_xx.bmm(V_x_)).split(Q_x.size(0), 0), 2)
        O_xy = torch.cat((Q_x_ + A_xy.bmm(V_y_)).split(Q_x.size(0), 0), 2)
        O_yx = torch.cat((Q_y_ + A_yx.bmm(V_x_)).split(Q_y.size(0), 0), 2)
        O_yy = torch.cat((Q_y_ + A_yy.bmm(V_y_)).split(Q_y.size(0), 0), 2)

        if getattr(self, 'ln0', None) is not None:
            O_xx = self.ln0(O_xx)
            O_xy = self.ln0(O_xy)
            O_yx = self.ln0(O_yx)
            O_yy = self.ln0(O_yy)

        O_x = F.relu(self.fc_X(torch.cat([O_xx,O_xy], dim=-1)))
        O_y = F.relu(self.fc_Y(torch.cat([O_yx,O_yy], dim=-1)))

        if getattr(self, 'ln1', None) is not None:
            O_x = self.ln1(O_x)
            O_y = self.ln1(O_y)

        return O_x, O_y


class CSAB3(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(CSAB3, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q_xx = nn.Linear(dim_Q, dim_V)
        self.fc_k_xx = nn.Linear(dim_K, dim_V)
        self.fc_v_xx = nn.Linear(dim_K, dim_V)
        self.fc_q_xy = nn.Linear(dim_Q, dim_V)
        self.fc_k_xy = nn.Linear(dim_K, dim_V)
        self.fc_v_xy = nn.Linear(dim_K, dim_V)
        self.fc_q_yx = nn.Linear(dim_Q, dim_V)
        self.fc_k_yx = nn.Linear(dim_K, dim_V)
        self.fc_v_yx = nn.Linear(dim_K, dim_V)
        self.fc_q_yy = nn.Linear(dim_Q, dim_V)
        self.fc_k_yy = nn.Linear(dim_K, dim_V)
        self.fc_v_yy = nn.Linear(dim_K, dim_V)
        self.fc_X = nn.Linear(dim_V*2, dim_V)
        self.fc_Y = nn.Linear(dim_V*2, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

    def forward(self, inputs):
        X, Y = inputs
        Q_xx = self.fc_q_x(X)
        K_xx, V_xx = self.fc_k_xx(X), self.fc_v_xx(X)
        Q_xy = self.fc_q_xy(X)
        K_xy, V_xy = self.fc_k_xy(Y), self.fc_v_xy(Y)
        Q_yx = self.fc_q_yx(Y)
        K_yx, V_yx = self.fc_k_yx(X), self.fc_v_yx(X)
        Q_yy = self.fc_q_yy(Y)
        K_yy, V_yy = self.fc_k_yy(Y), self.fc_v_yy(Y)

        dim_split = self.dim_V // self.num_heads
        Q_xx_ = torch.cat(Q_xx.split(dim_split, 2), 0)
        K_xx_ = torch.cat(K_xx.split(dim_split, 2), 0)
        V_xx_ = torch.cat(V_xx.split(dim_split, 2), 0)
        Q_xy_ = torch.cat(Q_xy.split(dim_split, 2), 0)
        K_xy_ = torch.cat(K_xy.split(dim_split, 2), 0)
        V_xy_ = torch.cat(V_xy.split(dim_split, 2), 0)
        Q_yx_ = torch.cat(Q_yx.split(dim_split, 2), 0)
        K_yx_ = torch.cat(K_yx.split(dim_split, 2), 0)
        V_yx_ = torch.cat(V_yx.split(dim_split, 2), 0)
        Q_yy_ = torch.cat(Q_yy.split(dim_split, 2), 0)
        K_yy_ = torch.cat(K_yy.split(dim_split, 2), 0)
        V_yy_ = torch.cat(V_yy.split(dim_split, 2), 0)

        E_xx = Q_xx_.bmm(K_xx_.transpose(1,2))
        E_xy = Q_xy_.bmm(K_xy_.transpose(1,2))
        E_yx = Q_yx_.bmm(K_yx_.transpose(1,2))
        E_yy = Q_yy_.bmm(K_yy_.transpose(1,2))
        E_x = torch.cat([E_xx, E_xy], dim=1)
        E_y = torch.cat([E_yx, E_yy], dim=1)

        A_x = torch.softmax(E_x/math.sqrt(self.dim_V), 2)
        A_y = torch.softmax(E_y/math.sqrt(self.dim_V), 2)

        O_x = torch.cat((torch.cat([Q_xx_, Q_xy_], dim=1) + A_x.bmm(torch.cat([V_xx_,V_xy_],1))).split(Q_xx.size(0), 0), 2)
        O_y = torch.cat((torch.cat([Q_yx_, Q_yy_], dim=1) + A_y.bmm(torch.cat([V_yx_,V_yy_],1))).split(Q_yx.size(0), 0), 2)

        if getattr(self, 'ln0', None) is not None:
            O_x = self.ln0(O_x)
            O_y = self.ln0(O_y)

        O_x = O_x + F.relu(self.fc_X(O_x))
        O_y = O_y + F.relu(self.fc_Y(O_y))

        if getattr(self, 'ln1', None) is not None:
            O_x = self.ln1(O_x)
            O_y = self.ln1(O_y)

        return O_x, O_y

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze(-1)

class MultiSetTransformer1(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, num_blocks=2, ln=False):
        super(MultiSetTransformer1, self).__init__()
        self.proj = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(*[CSAB(dim_hidden, dim_hidden, num_heads, ln=ln) for i in range(num_blocks)])
        self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(2*dim_hidden, dim_output),)

    def forward(self, X, Y):
        ZX, ZY = self.enc((self.proj(X),self.proj(Y)))
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)

class MultiSetTransformer2(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, num_blocks=2, ln=False):
        super(MultiSetTransformer2, self).__init__()
        self.proj = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(*[CSAB2(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln) for i in range(num_blocks)])
        self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden*2, dim_output))

    def forward(self, X, Y):
        ZX, ZY = self.enc((self.proj(X),self.proj(Y)))
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)

class MultiSetTransformer3(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, num_blocks=2, ln=False):
        super(MultiSetTransformer2, self).__init__()
        self.proj = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(*[CSAB3(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln) for i in range(num_blocks)])
        self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden*2, dim_output))

    def forward(self, X, Y):
        ZX, ZY = self.enc((self.proj(X),self.proj(Y)))
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)


class EquiSetTransformer1(nn.Module):
    def __init__(self, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
                EquiSAB1(num_heads, ln=ln),
                EquiSAB1(num_heads, ln=ln))
        self.pool = EquiEncoder(dim_hidden, input_size=1)
        self.dec = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        Z = self.enc(X).sum(dim=1)
        Z = self.pool(Z)
        return self.dec(Z).squeeze(-1)

class EquiSetTransformer2(nn.Module):
    def __init__(self, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
                EquiSAB2(dim_hidden, num_heads, ln=ln),
                EquiSAB2(dim_hidden, num_heads, ln=ln))
        self.dec = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        Z = self.enc(X).sum(dim=1)
        return self.dec(Z).squeeze(-1)

class EquiSetTransformer3(nn.Module):
    def __init__(self, dim_output, num_outputs=1,
            dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
                EquiSAB3(1, dim_hidden, num_heads, ln=ln),
                EquiSAB3(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        Z = self.enc(X.unsqueeze(-1))
        Z = Z.max(dim=2)[0]
        return self.dec(Z).squeeze(-1)

'''
class EquiMultiSetTransformer1(nn.Module):
    def __init__(self, dim_output, dim_hidden=128, num_heads=4, num_blocks=2, ln=False):
        super().__init__()
        self.enc = nn.Sequential(*[EquiCSAB(num_heads, ln=ln) for i in range(num_blocks)])
        #self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        #self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool = EquiEncoder(dim_hidden, input_size=2)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(2*dim_hidden, dim_output),)

    def forward(self, X, Y):
        ZX, ZY = self.enc((X,Y))
        ZX = torch.sum(ZX, dim=1)
        ZY = torch.sum(ZY, dim=1)
        Z = self.pool(torch.cat([ZX.unsqueeze(-1), ZY.unsqueeze(-1)], dim=-1))
        return self.dec(Z).squeeze(-1)
'''

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
        ZX = ZX.max(dim=2)[0]
        ZY = ZY.max(dim=2)[0]
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)