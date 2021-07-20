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
        Z_XX = self.encoder1(X, X)
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


#
#   From Set Transformers
#

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

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

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
        K_x, V_x = self.fc_k_x(X), self.fc_v(X)
        Q_y = self.fc_q_x(Y)
        K_y, V_y = self.fc_k_x(Y), self.fc_v(Y)

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

        O_x = F.relu(self.fc_X(torch.cat([O_xx,O_xy])))
        O_y = F.relu(self.fc_Y(torch.cat([O_yx,O_yy])))

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

    def forward(self, X, Y):
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
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(MultiSetTransformer1, self).__init__()
        self.proj = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(
                CSAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                CSAB(dim_hidden, dim_hidden, num_heads, ln=ln))
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
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(MultiSetTransformer2, self).__init__()
        self.enc = nn.Sequential(
                CSAB2(dim_input, dim_input, dim_hidden, num_heads, ln=ln),
                CSAB2(dim_hidden, dim_hidden, dim_hidden, num_heads, ln=ln))
        self.pool_x = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.pool_y = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden*2, dim_output))

    def forward(self, X, Y):
        ZX, ZY = self.enc((X,Y))
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        return self.dec(torch.cat([ZX, ZY], dim=-1)).squeeze(-1)

class MultiSetTransformer3(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(MultiSetTransformer3, self).__init__()
        self.enc = nn.Sequential(
                SAB(dim_input+1, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def prepare_inputs(self, X, Y):
        # each should be size batch_size x n_elements x dim
        X = torch.cat([X, torch.zeros(*X.size()[:-1], 1)], dim=-1)
        Y = torch.cat([Y, torch.ones(*Y.size()[:-1], 1)], dim=-1)
        return torch.cat([X,Y], dim=1)

    def forward(self, X, Y):
        return self.dec(self.enc(self.prepare_inputs(X,Y)))

class MultiSetTransformer4(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(MultiSetTransformer4, self).__init__()
        self.X_encoding = nn.Parameter(torch.empty(dim_input))
        self.Y_encoding = nn.Parameter(torch.empty(dim_input))
        self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

        nn.init.uniform_(self.X_encoding, math.sqrt(5))
        nn.init.uniform_(self.Y_encoding, math.sqrt(5))

    def forward(self, X, Y):
        inputs = torch.cat([X+self.X_encoding, Y+self.Y_encoding], dim=1)
        return self.dec(self.enc(inputs))

