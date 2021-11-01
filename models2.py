import torch
import torch.nn as nn
import math

use_cuda=torch.cuda.is_available()

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

class MHA(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads,):
        super(MHA, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_Q, dim_V, bias=True)
        self.w_k = nn.Linear(dim_K, dim_V, bias=True)
        self.w_v = nn.Linear(dim_K, dim_V, bias=True)
        self.w_o = nn.Linear(dim_V, dim_V, bias=True)

    def forward(self, Q, K, mask=None):
        Q_ = self.w_q(Q)
        K_, V_ = self.w_k(K), self.w_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.stack(Q_.split(dim_split, 2), 0)
        K_ = torch.stack(K_.split(dim_split, 2), 0)
        V_ = torch.stack(V_.split(dim_split, 2), 0)

        E = Q_.matmul(K_.transpose(2,3))/math.sqrt(self.dim_V)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_)).split(1, 0), 3).squeeze(0))
        return O

class EquiMHA(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super(EquiMHA, self).__init__()
        self.latent_size=dim_V
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_Q, dim_V, bias=False)
        self.w_k = nn.Linear(dim_K, dim_V, bias=False)
        self.w_v = nn.Linear(dim_K, dim_V, bias=False)
        self.w_o = nn.Linear(dim_V, dim_V, bias=False)
    
    def forward(self, Q, K, mask=None):
        Q = self.w_q(Q)
        K, V = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q.split(dim_split, 3), 0)
        K_ = torch.stack(K.split(dim_split, 3), 0)
        V_ = torch.stack(V.split(dim_split, 3), 0)

        E = Q_.transpose(2,3).matmul(K_.transpose(2,3).transpose(3,4)).sum(dim=2) / math.sqrt(self.latent_size)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_.view(*V_.size()[:-2], -1)).view(*Q_.size())).split(1, 0), 4).squeeze(0))
        return O

class MAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, attn_size=None, ln=False, equi=False, dropout=0.1):
        super(MAB, self).__init__()
        attn_size = attn_size if attn_size is not None else input_size
        if equi:
            self.attn = EquiMHA(input_size, attn_size, latent_size, num_heads)
        else:
            self.attn = MHA(input_size, attn_size, latent_size, num_heads)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)

    def forward(self, Q, K, mask=None):
        X = Q + self.attn(Q, K, mask=mask)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln0', None) is None else self.ln0(X)
        X = X + self.fc(X)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln1', None) is None else self.ln1(X)
        return X

class SAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, ln=False, remove_diag=False, equi=False):
        super(SAB, self).__init__()
        self.mab = MAB(input_size, latent_size, hidden_size, num_heads, ln=False, remove_diag=False, equi=False)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask=mask)

class ISAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, num_inds, ln=False, equi=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, latent_size))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(latent_size, latent_size, hidden_size, num_heads, attn_size=input_size, ln=ln, equi=equi)
        self.mab1 = MAB(input_size, latent_size, hidden_size, num_heads, ln=ln, equi=equi)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class CSAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, remove_diag=False, **kwargs):
        super(CSAB, self).__init__()
        self.MAB_XX = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB_YY = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB_XY = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB_YX = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.fc_X = nn.Linear(latent_size * 2, latent_size)
        self.fc_Y = nn.Linear(latent_size * 2, latent_size)
        self.remove_diag = remove_diag

    def _get_masks(self, N, M, masks):
        if self.remove_diag:
            diag_xx = (1 - torch.eye(N)).unsqueeze(0)
            diag_yy = (1 - torch.eye(M)).unsqueeze(0)
            if use_cuda:
                diag_xx = diag_xx.cuda()
                diag_yy = diag_yy.cuda()
            if masks is not None:
                mask_xx, mask_xy, mask_yx, mask_yy = masks
                mask_xx = mask_xx * diag_xx
                mask_yy = mask_yy * diag_yy
            else:
                mask_xx, mask_yy = diag_xx, diag_yy
                mask_xy, mask_yx = None, None
        else:
            if masks is not None:
                mask_xx, mask_xy, mask_yx, mask_yy = masks 
            else: 
                mask_xx, mask_xy, mask_yx, mask_yy = None,None,None,None
        return mask_xx, mask_xy, mask_yx, mask_yy

    def forward(self, inputs, masks=None):
        X, Y = inputs
        mask_xx, mask_xy, mask_yx, mask_yy = self._get_masks(X.size(1), Y.size(1), masks)
        XX = self.MAB_XX(X, X, mask=mask_xx)
        XY = self.MAB_XY(X, Y, mask=mask_xy)
        YX = self.MAB_YX(Y, X, mask=mask_yx)
        YY = self.MAB_YY(Y, Y, mask=mask_yy)
        X_out = X + self.fc_X(torch.cat([XX, XY], dim=-1))
        Y_out = Y + self.fc_Y(torch.cat([YY, YX], dim=-1))
        return (X_out, Y_out)


class ICSAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, num_inds, **kwargs):
        super(ICSAB, self).__init__()
        self.I_X = nn.Parameter(torch.Tensor(1, num_inds, latent_size))
        self.I_Y = nn.Parameter(torch.Tensor(1, num_inds, latent_size))
        nn.init.xavier_uniform_(self.I_X)
        nn.init.xavier_uniform_(self.I_Y)
        self.MAB0_X = MAB(latent_size, latent_size, hidden_size, num_heads, attn_size=input_size, **kwargs)
        self.MAB0_Y = MAB(latent_size, latent_size, hidden_size, num_heads, attn_size=input_size, **kwargs)
        #self.MAB0_XY = MAB(latent_size, latent_size, hidden_size, num_heads, attn_size=input_size, equi=equi, ln=ln)
        #self.MAB0_YX = MAB(latent_size, latent_size, hidden_size, num_heads, attn_size=input_size, equi=equi, ln=ln)
        self.MAB1_XX = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB1_YY = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB1_XY = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.MAB1_YX = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        self.fc_X = nn.Linear(latent_size * 2, latent_size)
        self.fc_Y = nn.Linear(latent_size * 2, latent_size)

    def forward(self, inputs, masks=None):
        X,Y = inputs
        if self.equi:
            I_X = self.I_X.unsqueeze(-2).expand(X.size(0), -1, X.size(2), -1)
            I_Y = self.I_Y.unsqueeze(-2).expand(Y.size(0), -1, Y.size(2), -1)
        else:
            I_X, I_Y = self.I_X.expand(X.size(0), 1, 1), self.I_Y.expand(Y.size(0), 1, 1)
        H_X = self.MAB0_X(I_X, X)
        H_Y = self.MAB0_Y(I_Y, Y)
        XX = self.MAB1_XX(X, H_X)
        XY = self.MAB1_XY(X, H_Y)
        YX = self.MAB1_YX(Y, H_X)
        YY = self.MAB1_YY(Y, H_Y)
        X_out = X + self.fc_X(torch.cat([XX, XY], dim=-1))
        Y_out = Y + self.fc_Y(torch.cat([YY, YX], dim=-1))
        return (X_out, Y_out)





class PMA(nn.Module):
    def __init__(self, latent_size, hidden_size, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, latent_size))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(latent_size, latent_size, hidden_size, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class EncoderStack(nn.Sequential):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

class SetTransformer(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=2, remove_diag=False, ln=False, equi=False):
        super(SetTransformer, self).__init__()
        if equi:
            input_size = 1
        self.proj = nn.Linear(input_size, latent_size)
        self.enc = nn.Sequential(*[SAB(input_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi) for _ in range(num_blocks)])
        self.pool = PMA(latent_size, num_heads, 1, ln=ln)
        self.dec = nn.Linear(latent_size, output_size)
                
    def forward(self, X):
        Xproj = self.proj(X.unsqueeze(-1)) if self.equi else self.proj(X)
        ZX = self.enc(Xproj)
        if self.equi:
            ZX = ZX.max(dim=2)[0]
        ZX = self.pool_x(ZX)
        return self.dec(ZX).squeeze(-1)

class MultiSetTransformer(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=2, remove_diag=False, ln=False, equi=False, dropout=0.1, num_inds=-1):
        super(MultiSetTransformer, self).__init__()
        if equi:
            input_size = 1
        self.proj = nn.Linear(input_size, latent_size)
        if num_inds > 0:
            self.enc = EncoderStack(*[ICSAB(latent_size, latent_size, hidden_size, num_heads, num_inds, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for i in range(num_blocks)])
        else:
            self.enc = EncoderStack(*[CSAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for i in range(num_blocks)])
        self.pool_x = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.pool_y = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.dec = nn.Sequential(
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                #SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(2*latent_size, output_size),)
        self.remove_diag = remove_diag
        self.equi=equi

    def forward(self, X, Y, masks=None):
        if self.equi:
            Xproj, Yproj = self.proj(X.unsqueeze(-1)), self.proj(Y.unsqueeze(-1))
        else:
            Xproj, Yproj = self.proj(X), self.proj(Y)
        ZX, ZY = self.enc((Xproj, Yproj), masks=masks)
        if self.equi:
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]
        ZX = self.pool_x(ZX)
        ZY = self.pool_y(ZY)
        out = self.dec(torch.cat([ZX, ZY], dim=-1))
        return out.squeeze(-1)





#
#   PINE
#

import torch.nn.functional as F
class PINE(nn.Module):
    def __init__(self, input_size, proj_size, n_proj, n_sets, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.proj_size = proj_size
        self.n_proj = n_proj
        self.n_sets = n_sets
        for i in range(n_sets):
            self.register_parameter('U_%d'%i, nn.Parameter(torch.empty(n_proj, proj_size, 1)))
            self.register_parameter('A_%d'%i, nn.Parameter(torch.empty(n_proj, 1, input_size)))
            self.register_parameter('V_%d'%i, nn.Parameter(torch.empty(n_proj * proj_size)))
        self.W_h = nn.Parameter(torch.empty(hidden_size, n_sets * n_proj * proj_size))
        self.C = nn.Linear(hidden_size, output_size)

        self._init_params()

    def _init_params(self):
        for i in range(self.n_sets):
            nn.init.kaiming_uniform_(getattr(self,'U_%d'%i), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self,'A_%d'%i), a=math.sqrt(5))
            W_g_i = torch.matmul(getattr(self,'U_%d'%i), getattr(self,'A_%d'%i))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W_g_i)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(getattr(self,'V_%d'%i), -bound, bound)
        nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))

    def forward(self, *X):
        #assume X is a list of tensors of size bs x n_k x d each
        z = []
        for i in range(self.n_sets):
            W_g_i = torch.matmul(getattr(self,'U_%d'%i), getattr(self,'A_%d'%i)).view(-1, self.input_size)
            g = torch.sigmoid(X[i].matmul(W_g_i.transpose(-1,-2)) + getattr(self,'V_%d'%i))
            z.append(g.sum(dim=1))
        z_stacked = torch.cat(z, dim=-1)
        h = torch.sigmoid(z_stacked.matmul(self.W_h.t()))
        return self.C(h)

class EquiPINE(nn.Module):
    def __init__(self, latent_size, proj_size, n_proj, n_sets, hidden_size, output_size):
        super().__init__()
        self.latent_size = latent_size
        self.proj_size = proj_size
        self.n_proj = n_proj
        self.n_sets = n_sets
        for i in range(n_sets):
            self.register_parameter('P_%d'%i, nn.Parameter(torch.empty(latent_size, 1)))
            self.register_parameter('U_%d'%i, nn.Parameter(torch.empty(n_proj, proj_size, 1)))
            self.register_parameter('A_%d'%i, nn.Parameter(torch.empty(n_proj, 1, latent_size)))
            self.register_parameter('V_%d'%i, nn.Parameter(torch.empty(n_proj * proj_size)))
        self.W_h = nn.Parameter(torch.empty(hidden_size, n_sets * n_proj * proj_size))
        self.C = nn.Linear(hidden_size, output_size)

        self._init_params()

    def _init_params(self):
        for i in range(self.n_sets):
            nn.init.kaiming_uniform_(getattr(self,'U_%d'%i), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self,'A_%d'%i), a=math.sqrt(5))
            W_g_i = torch.matmul(getattr(self,'U_%d'%i), getattr(self,'A_%d'%i))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(W_g_i)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(getattr(self,'V_%d'%i), -bound, bound)
        nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))

    def forward(self, *X):
        #assume X is a list of tensors of size bs x n_k x d each
        z = []
        for i in range(self.n_sets):
            X_i = torch.matmul(X[i].unsqueeze(-1), getattr(self, 'P_%d'%i).transpose(0,1))
            W_g_i = torch.matmul(getattr(self,'U_%d'%i), getattr(self,'A_%d'%i)).view(-1, self.latent_size)
            g = torch.sigmoid(X_i.matmul(W_g_i.transpose(-1,-2)) + getattr(self,'V_%d'%i))
            z.append(g.max(dim=2).sum(dim=1))
        z_stacked = torch.cat(z, dim=-1)
        h = torch.sigmoid(z_stacked.matmul(self.W_h.t()))
        return self.C(h)


#
#   RN
#

class RelationNetwork(nn.Module):
    def __init__(self, net, remove_diag=False, pool='sum'):
        super().__init__()
        self.net = net
        self.remove_diag = remove_diag
        self.pool = pool
    
    def forward(self, X, Y):
        N = X.size(1)
        M = Y.size(1)
        pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,*Y.size()[2:]), X.unsqueeze(2).expand(-1,-1,M,*X.size()[2:])], dim=-1)
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

class RNBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, remove_diag=False, ln=False, pool='sum'):
        super().__init__()
        net = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        self.rn = RelationNetwork(net, remove_diag, pool)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)

    def forward(self, X):
        Z = X + self.rn(X)
        Z = Z if getattr(self, 'ln0', None) is None else self.ln0(Z)
        Z = Z + self.fc(Z)
        Z = Z if getattr(self, 'ln1', None) is None else self.ln1(Z)
        return Z

class MultiRNBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, remove_diag=False, pool='sum', ln=False):
        super().__init__()
        self.e_xx = RNBlock(latent_size, hidden_size, remove_diag=remove_diag, pool=pool, ln=ln)
        self.e_xy = RNBlock(latent_size, hidden_size, remove_diag=False, pool=pool, ln=ln)
        self.e_yx = RNBlock(latent_size, hidden_size, remove_diag=False, pool=pool, ln=ln)
        self.e_yy = RNBlock(latent_size, hidden_size, remove_diag=remove_diag, pool=pool, ln=ln)
        self.fc_X = nn.Linear(2*latent_size, latent_size)
        self.fc_Y = nn.Linear(2*latent_size, latent_size)
        self.remove_diag = remove_diag
        self.pool = pool
    
    def forward(self, X, Y):
        Z_XX = self.e_xx(X, X)
        Z_XY = self.e_xy(X, Y)
        Z_YX = self.e_yx(Y, X)
        Z_YY = self.e_yy(Y, Y)
        X_out = X + F.relu(self.fc_X(torch.cat([Z_XX, Z_XY], dim=-1)))
        Y_out = X + F.relu(self.fc_Y(torch.cat([Z_YY, Z_YX], dim=-1)))

        return X_out, Y_out


class RNModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks=2, remove_diag=False, ln=False, pool1='sum', pool2='sum', equi=False):
        super().__init__()
        if equi:
            input_size = 1
        self.proj = nn.Linear(input_size, latent_size)
        self.enc = nn.Sequential(*[RNBlock(latent_size, hidden_size, ln=ln, remove_diag=remove_diag, pool=pool1) for _ in range(num_blocks)])
        self.dec = nn.Linear(latent_size, output_size)
                
    def forward(self, X):
        Xproj = self.proj(X.unsqueeze(-1)) if self.equi else self.proj(X)
        ZX = self.enc(Xproj)
        if self.equi:
            ZX = ZX.max(dim=2)[0]
        if self.pool == 'sum':
            ZX = torch.sum(ZX, dim=1)
        elif self.pool == 'max':
            ZX = torch.max(ZX, dim=1)[0]
        else:
            raise NotImplementedError()
        return self.dec(ZX).squeeze(-1)

class MultiRNModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks=2, remove_diag=False, ln=False, pool1='sum', pool2='sum', equi=False):
        super().__init__()
        if equi:
            input_size = 1
        self.proj = nn.Linear(input_size, latent_size)
        self.enc = EncoderStack(*[MultiRNBlock(latent_size, hidden_size, ln=ln, remove_diag=remove_diag, pool=pool1) for i in range(num_blocks)])
        self.dec = nn.Linear(2*latent_size, output_size)
        self.remove_diag = remove_diag
        self.equi=equi

    def forward(self, X, Y, masks=None):
        if self.equi:
            Xproj, Yproj = self.proj(X.unsqueeze(-1)), self.proj(Y.unsqueeze(-1))
        else:
            Xproj, Yproj = self.proj(X), self.proj(Y)
        ZX, ZY = self.enc((Xproj, Yproj), masks=masks)
        if self.equi:
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]
        ZX = ZX.max(dim=1)[0]
        ZY = ZY.max(dim=1)[0]
        out = self.dec(torch.cat([ZX, ZY], dim=-1))
        return out.squeeze(-1)

