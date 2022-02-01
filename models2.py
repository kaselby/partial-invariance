import torch
import torch.nn as nn
import math

from utils import cross_knn_inds

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
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, bias=None, equi=False, nn_attn=False):
        super(MHA, self).__init__()
        if bias is None:
            bias = not equi
        self.latent_size = dim_V
        self.num_heads = num_heads
        self.w_q = nn.Linear(dim_Q, dim_V, bias=bias)
        self.w_k = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_v = nn.Linear(dim_K, dim_V, bias=bias)
        self.w_o = nn.Linear(dim_V, dim_V, bias=bias)
        self.equi = equi
        self.nn_attn = nn_attn

    def _mha(self, Q, K, mask=None):
        Q_ = self.w_q(Q)
        K_, V_ = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q_.split(dim_split, 2), 0)
        K_ = torch.stack(K_.split(dim_split, 2), 0)
        V_ = torch.stack(V_.split(dim_split, 2), 0)

        E = Q_.matmul(K_.transpose(2,3))/math.sqrt(self.latent_size)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_)).split(1, 0), 3).squeeze(0))
        return O
    
    def _equi_mha(self, Q, K, mask=None):
        # band-aid fix for backwards compat:
        d = self.latent_size if getattr(self, 'latent_size', None) is not None else self.dim_V

        Q = self.w_q(Q)
        K, V = self.w_k(K), self.w_v(K)

        dim_split = d // self.num_heads
        Q_ = torch.stack(Q.split(dim_split, 3), 0)
        K_ = torch.stack(K.split(dim_split, 3), 0)
        V_ = torch.stack(V.split(dim_split, 3), 0)

        E = Q_.transpose(2,3).matmul(K_.transpose(2,3).transpose(3,4)).sum(dim=2) / math.sqrt(d)
        if mask is not None:
            A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
        else:
            A = torch.softmax(E, 3)
        O = self.w_o(torch.cat((A.matmul(V_.view(*V_.size()[:-2], -1)).view(*Q_.size())).split(1, 0), 4).squeeze(0))
        return O

    def _nn_mha(self, Q, K, neighbours):
        N,M = Q.size(1),K.size(1)

        Q_ = self.w_q(Q)
        K_, V_ = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q_.split(dim_split, 2), 0)
        K_ = torch.stack(K_.split(dim_split, 2), 0)
        V_ = torch.stack(V_.split(dim_split, 2), 0)

        K_neighbours = torch.gather(K_.unsqueeze(2).expand(-1,-1, N,-1,-1), 3, neighbours.unsqueeze(-1).unsqueeze(0).expand(self.num_heads,-1,-1,-1,dim_split))
        V_neighbours = torch.gather(V_.unsqueeze(2).expand(-1,-1, N,-1,-1), 3, neighbours.unsqueeze(-1).unsqueeze(0).expand(self.num_heads,-1,-1,-1,dim_split))
        E = Q_.unsqueeze(3).matmul(K_neighbours.transpose(3,4)).squeeze(3)/math.sqrt(self.latent_size)
        A = torch.softmax(E, 3)
        AV = A.unsqueeze(3).matmul(V_neighbours).squeeze(3)

        O = self.w_o(torch.cat((AV).split(1, 0), 3).squeeze(0))
        return O

    def _nn_equi_mha(self, Q, K, neighbours):
        N,M = Q.size(1),K.size(1)
        bs = Q.size(0)
        k = neighbours.size(-1)
        d = Q.size(2)

        Q = self.w_q(Q)
        K, V = self.w_k(K), self.w_v(K)

        dim_split = self.latent_size // self.num_heads
        Q_ = torch.stack(Q.split(dim_split, 3), 0)
        K_ = torch.stack(K.split(dim_split, 3), 0)
        V_ = torch.stack(V.split(dim_split, 3), 0)

        K_neighbours = torch.gather(K_.transpose(2,3).unsqueeze(3).expand(-1,-1,-1,N,-1,-1), 4, neighbours.view(1, bs, 1, N, k, 1).expand(self.num_heads,-1,d,-1,-1,dim_split))
        V_neighbours = torch.gather(V_.unsqueeze(2).expand(-1,-1,N,-1,-1,-1), 3, neighbours.view(1, bs, N, k, 1, 1).expand(self.num_heads,-1, -1, -1, d,dim_split))
        E = Q_.transpose(2,3).unsqueeze(4).matmul(K_neighbours.transpose(4,5)).squeeze(4).sum(dim=2) / math.sqrt(self.latent_size)
        A = torch.softmax(E, 3)
        AV = A.unsqueeze(3).matmul(V_neighbours.view(*V_neighbours.size()[:-2], -1)).squeeze(3).view(*Q_.size())
        
        O = self.w_o(torch.cat((AV).split(1, 0), 4).squeeze(0))
        return O

    def forward(self, *args, **kwargs):
        if getattr(self, 'equi', False):
            if not getattr(self, 'nn_attn', False):
                return self._equi_mha(*args, **kwargs)
            else:
                return self._nn_equi_mha(*args, **kwargs)
        else:
            if not getattr(self, 'nn_attn', False):
                return self._mha(*args, **kwargs)
            else:
                return self._nn_mha(*args, **kwargs)
                


class MAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, attn_size=None, ln=False, equi=False, nn_attn=False, dropout=0.1):
        super(MAB, self).__init__()
        attn_size = attn_size if attn_size is not None else input_size
        self.attn = MHA(input_size, attn_size, latent_size, num_heads, equi=equi, nn_attn=nn_attn)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)

    def forward(self, Q, K, **kwargs):
        X = Q + self.attn(Q, K, **kwargs)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln0', None) is None else self.ln0(X)
        X = X + self.fc(X)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln1', None) is None else self.ln1(X)
        return X

class SAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, ln=False, remove_diag=False, equi=False, nn=False, dropout=0.1):
        super(SAB, self).__init__()
        self.mab = MAB(input_size, latent_size, hidden_size, num_heads, ln=ln, equi=equi, dropout=dropout)

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


class CSABSimple(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, weight_sharing='none', **kwargs):
        super(CSABSimple, self).__init__()
        self._init_blocks(input_size, latent_size, hidden_size, num_heads, weight_sharing, **kwargs)
        self.fc_X = nn.Linear(latent_size, latent_size)
        self.fc_Y = nn.Linear(latent_size, latent_size)

    def _init_blocks(self, input_size, latent_size, hidden_size, num_heads, weight_sharing='none', **kwargs):
        if weight_sharing == 'none':
            self.MAB_XY = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_YX = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
        else:
            MAB_cross = MAB(input_size, latent_size, hidden_size, num_heads, **kwargs)
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross

    def forward(self, inputs):
        X, Y = inputs
        XY = self.MAB_XY(X, Y)
        YX = self.MAB_YX(Y, X)
        X_out = X + self.fc_X(XY)
        Y_out = Y + self.fc_Y(YX)
        return (X_out, Y_out)

class CSAB(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, remove_diag=False, nn_attn=False, weight_sharing='none', merge='concat', **kwargs):
        super(CSAB, self).__init__()
        self._init_blocks(input_size, latent_size, hidden_size, num_heads, remove_diag, nn_attn, weight_sharing, **kwargs)
        self.merge = merge
        if self.merge == 'concat':
            self.fc_X = nn.Linear(latent_size * 2, latent_size)
            self.fc_Y = nn.Linear(latent_size * 2, latent_size)
        else:
            self.fc_X = nn.Linear(latent_size, latent_size)
            self.fc_Y = nn.Linear(latent_size, latent_size)
        self.remove_diag = remove_diag
        self.nn_attn = nn_attn

    def _init_blocks(self, input_size, latent_size, hidden_size, num_heads, remove_diag=False, nn_attn=False, weight_sharing='none', **kwargs):
        if weight_sharing == 'none':
            self.MAB_XX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_YY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_XY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_YX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
        elif weight_sharing == 'cross':
            self.MAB_XX = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_YY = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            MAB_cross = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross
        elif weight_sharing == 'sym':
            MAB_cross = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            MAB_self = MAB(input_size, latent_size, hidden_size, num_heads, nn_attn=nn_attn, **kwargs)
            self.MAB_XX = MAB_self
            self.MAB_YY = MAB_self
            self.MAB_XY = MAB_cross
            self.MAB_YX = MAB_cross
        else:
            raise NotImplementedError("weight sharing must be none, cross or sym")

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

    def forward(self, inputs, masks=None, neighbours=None):
        X, Y = inputs
        if not getattr(self, 'nn_attn', False):
            mask_xx, mask_xy, mask_yx, mask_yy = self._get_masks(X.size(1), Y.size(1), masks)
            XX = self.MAB_XX(X, X, mask=mask_xx)
            XY = self.MAB_XY(X, Y, mask=mask_xy)
            YX = self.MAB_YX(Y, X, mask=mask_yx)
            YY = self.MAB_YY(Y, Y, mask=mask_yy)
        else:
            assert neighbours is not None and masks is None
            N_XX, N_XY, N_YX, N_YY = neighbours
            XX = self.MAB_XX(X, X, neighbours=N_XX)
            XY = self.MAB_XY(X, Y, neighbours=N_XY)
            YX = self.MAB_YX(Y, X, neighbours=N_YX)
            YY = self.MAB_YY(Y, Y, neighbours=N_YY)
        #backwards compatibility
        if getattr(self, "merge", None) is None or self.merge == "concat":
            X_out = X + self.fc_X(torch.cat([XX, XY], dim=-1))
            Y_out = Y + self.fc_Y(torch.cat([YY, YX], dim=-1))
        else:
            X_out = X + self.fc_X(XX + XY)
            Y_out = Y + self.fc_Y(YX + YY)
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
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=2, remove_diag=False, ln=False, equi=False, dropout=0.1):
        super(SetTransformer, self).__init__()
        if equi:
            input_size = 1
        self.equi=equi
        self.proj = None if input_size == latent_size else nn.Linear(input_size, latent_size) 
        self.enc = nn.Sequential(*[SAB(input_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for _ in range(num_blocks)])
        self.pool = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.dec = nn.Linear(latent_size, output_size)
                
    def forward(self, X):
        ZX = X
        if self.equi:
            ZX= ZX.unsqueeze(-1)
        if self.proj is not None:
            ZX= self.proj(ZX)
        ZX = self.enc(ZX)
        if self.equi:
            ZX = ZX.max(dim=2)[0]
        ZX = self.pool(ZX)
        return self.dec(ZX).squeeze(-1)

class MultiSetTransformer(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_heads=4, num_blocks=2, remove_diag=False, ln=False, equi=False, 
            nn_attn=False, weight_sharing='none', k_neighbours=5, dropout=0.1, num_inds=-1, decoder_layers=0, pool='pma', merge='concat'):
        super(MultiSetTransformer, self).__init__()
        if equi:
            input_size = 1
        self.input_size = input_size
        self.proj = None if input_size == latent_size else nn.Linear(input_size, latent_size) 
        if num_inds > 0:
            self.enc = EncoderStack(*[ICSAB(latent_size, latent_size, hidden_size, num_heads, num_inds, ln=ln, remove_diag=remove_diag, 
                equi=equi, weight_sharing=weight_sharing, dropout=dropout) for i in range(num_blocks)])
        else:
            self.enc = EncoderStack(*[CSAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, 
                equi=equi, nn_attn=nn_attn, weight_sharing=weight_sharing, dropout=dropout, merge='concat') for i in range(num_blocks)])
        self.pool_method = pool
        if self.pool_method == "pma":
            self.pool_x = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
            self.pool_y = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.dec = self._make_decoder(latent_size, hidden_size, output_size, decoder_layers)
        self.remove_diag = remove_diag
        self.equi=equi
        self.nn_attn = nn_attn
        self.k_neighbours = k_neighbours

    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers):
        if n_layers == 0:
            return nn.Linear(2*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y, masks=None):
        ZX, ZY = X, Y
        if self.equi:
            ZX, ZY = ZX.unsqueeze(-1), ZY.unsqueeze(-1)
        if self.proj is not None:
            ZX, ZY = self.proj(ZX), self.proj(ZY)
            
        if not getattr(self, "nn_attn", False):
            ZX, ZY = self.enc((ZX, ZY), masks=masks)
        else:
            neighbours = cross_knn_inds(X, Y, self.k_neighbours)
            ZX, ZY = self.enc((ZX, ZY), neighbours=neighbours)
            
        if self.equi:
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]
        
        #backwards compatibility
        if getattr(self, "pool_method", None) is None or self.pool_method == "pma":
            ZX = self.pool_x(ZX)
            ZY = self.pool_y(ZY)
        elif self.pool_method == "max":
            ZX = torch.max(ZX, dim=1)
            ZY = torch.max(ZY, dim=1)
        elif self.pool_method == "mean":
            ZX = torch.mean(ZX, dim=1)
            ZY = torch.mean(ZY, dim=1)

        out = self.dec(torch.cat([ZX, ZY], dim=-1))
        return out.squeeze(-1)


class NaiveMultiSetModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks, num_heads, remove_diag=False, ln=False,
            equi=False, weight_sharing='none', dropout=0.1, decoder_layers=1):
        super().__init__()
        self.equi = equi
        if equi:
            input_size = 1
        self.input_size = input_size
        self.proj = None if input_size == latent_size else nn.Linear(input_size, latent_size)
        if weight_sharing == 'none':
            self.encoder1 = nn.Sequential(*[SAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for _ in range(num_blocks)])
            self.encoder2 = nn.Sequential(*[SAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for _ in range(num_blocks)])
            self.pool1 = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
            self.pool2 = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
            #self.encoder1 = SetTransformer(input_size, latent_size, hidden_size, latent_size, num_heads, num_blocks, remove_diag, ln, equi)
            #self.encoder2 = SetTransformer(input_size, latent_size, hidden_size, latent_size, num_heads, num_blocks, remove_diag, ln, equi)
        else:
            #encoder = SetTransformer(input_size, latent_size, hidden_size, latent_size, num_heads, num_blocks, remove_diag, ln, equi)
            encoder = nn.Sequential(*[SAB(latent_size, latent_size, hidden_size, num_heads, ln=ln, remove_diag=remove_diag, equi=equi, dropout=dropout) for _ in range(num_blocks)])
            pool = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
            self.encoder1 = encoder
            self.encoder2 = encoder
            self.pool1 = pool
            self.pool2 = pool
        self.decoder = self._make_decoder(latent_size, hidden_size, output_size, decoder_layers)

    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers):
        if n_layers == 0:
            return nn.Linear(2*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y):
        ZX, ZY = X, Y
        if getattr(self, 'equi', False):
            ZX, ZY = ZX.unsqueeze(-1), ZY.unsqueeze(-1)
        if self.proj is not None:
            ZX, ZY = self.proj(ZX), self.proj(ZY)
        ZX = self.encoder1(ZX)
        ZY = self.encoder2(ZY)

        if getattr(self, 'equi', False):
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]

        ZX = self.pool1(ZX)
        ZY = self.pool2(ZY)
        out = self.decoder(torch.cat([ZX, ZY], dim=-1))
        return out.squeeze(-1)


class CrossOnlyModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks, num_heads, ln=False, 
            equi=False, weight_sharing='none', dropout=0.1, decoder_layers=1):
        super().__init__()
        if equi:
            input_size=1
        self.input_size = input_size
        self.equi = equi
        self.encoder = EncoderStack(*[CSABSimple(latent_size, latent_size, hidden_size, num_heads, ln=ln, equi=equi, 
            weight_sharing=weight_sharing, dropout=dropout) for i in range(num_blocks)])
        self.decoder = self._make_decoder(latent_size, hidden_size, output_size, decoder_layers)
        self.proj = None if input_size == latent_size else nn.Linear(input_size, latent_size)
        self.pool_x = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)
        self.pool_y = PMA(latent_size, hidden_size, num_heads, 1, ln=ln)

    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers):
        if n_layers == 0:
            return nn.Linear(2*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

    def forward(self, X, Y):
        ZX, ZY = X, Y
        if self.equi:
            ZX, ZY = ZX.unsqueeze(-1), ZY.unsqueeze(-1)
        if self.proj is not None:
            ZX, ZY = self.proj(ZX), self.proj(ZY)
        ZX, ZY = self.encoder((ZX, ZY))
        if self.equi:
            ZX = ZX.max(dim=2)[0]
            ZY = ZY.max(dim=2)[0]
        ZX = self.pool_x(ZX)
        ZY = self.pool_x(ZY)
        out = self.decoder(torch.cat([ZX, ZY], dim=-1))
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
    def __init__(self, net, pool='sum', equi=False):
        super().__init__()
        self.net = net
        self.pool = pool
        self.equi=equi
    
    def forward(self, X, Y, mask=None):
        N = X.size(1)
        M = Y.size(1)
        if self.equi:
            pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,-1,*Y.size()[3:]), X.unsqueeze(2).expand(-1,-1, M,-1,*X.size()[3:])], dim=-1)
        else:
            pairs = torch.cat([Y.unsqueeze(1).expand(-1,N,-1,*Y.size()[2:]), X.unsqueeze(2).expand(-1,-1,M,*X.size()[2:])], dim=-1)
        Z = self.net(pairs)
        if self.pool == 'sum':
            if mask is not None:
                if self.equi:
                    mask = mask.unsqueeze(-1)
                Z = Z * mask.unsqueeze(-1).expand_as(Z)
            Z = torch.sum(Z, dim=2)
        elif self.pool == 'max':
            if mask is not None:
                if self.equi:
                    mask = mask.unsqueeze(-1)
                Z = Z + mask.unsqueeze(-1).expand_as(Z) * -99999999
            Z = torch.max(Z, dim=2)[0]
        else:
            raise NotImplementedError()
        return Z

class RNBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, ln=False, pool='sum', dropout=0.1, equi=False):
        super().__init__()
        net = nn.Sequential(nn.Linear(2*latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        self.rn = RelationNetwork(net, pool, equi)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size)) 
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)

    def forward(self, X, Y, mask=None):
        Z = X + self.rn(X, Y, mask=mask)
        Z = Z if getattr(self, 'dropout', None) is None else self.dropout(Z)
        Z = Z if getattr(self, 'ln0', None) is None else self.ln0(Z)
        Z = Z + self.fc(Z)
        Z = Z if getattr(self, 'dropout', None) is None else self.dropout(Z)
        Z = Z if getattr(self, 'ln1', None) is None else self.ln1(Z)
        return Z

class MultiRNBlock(nn.Module):
    def __init__(self, latent_size, hidden_size, remove_diag=False, pool='max', ln=False, weight_sharing='none', **kwargs):
        super().__init__()
        self._init_blocks(latent_size, hidden_size, weight_sharing=weight_sharing, ln=ln, pool=pool, **kwargs)
        self.fc_X = nn.Linear(2*latent_size, latent_size)
        self.fc_Y = nn.Linear(2*latent_size, latent_size)
        self.remove_diag = remove_diag
        self.pool = pool

    def _init_blocks(self, latent_size, hidden_size, weight_sharing='none', ln=False, pool='max', **kwargs):
        if weight_sharing == 'none':
            self.e_xx = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_xy = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_yx = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_yy = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
        elif weight_sharing == 'cross':
            self.e_xx = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_yy = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            e_cross = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_xy = e_cross
            self.e_yx = e_cross
        elif weight_sharing == 'sym':
            e_cross = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            e_self = RNBlock(latent_size, hidden_size, pool=pool, ln=ln, **kwargs)
            self.e_xx = e_self
            self.e_yy = e_self
            self.e_xy = e_cross
            self.e_yx = e_cross
        else:
            raise NotImplementedError("weight sharing must be none, cross or sym")

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
        Z_XX = self.e_xx(X, X, mask=mask_xx)
        Z_XY = self.e_xy(X, Y, mask=mask_xy)
        Z_YX = self.e_yx(Y, X, mask=mask_yx)
        Z_YY = self.e_yy(Y, Y, mask=mask_yy)
        X_out = X + F.relu(self.fc_X(torch.cat([Z_XX, Z_XY], dim=-1)))
        Y_out = Y + F.relu(self.fc_Y(torch.cat([Z_YY, Z_YX], dim=-1)))

        return X_out, Y_out


class RNModel(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks=2, remove_diag=False, ln=False, pool1='sum', pool2='sum', equi=False, dropout=0.1):
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
    def __init__(self, input_size, latent_size, hidden_size, output_size, num_blocks=2, remove_diag=False, ln=False, 
            pool1='sum', pool2='sum', equi=False, dropout=0.1, weight_sharing='none', decoder_layers=0):
        super().__init__()
        if equi:
            input_size = 1
        self.input_size=input_size
        self.proj = nn.Linear(input_size, latent_size)
        self.enc = EncoderStack(*[MultiRNBlock(latent_size, hidden_size, ln=ln, remove_diag=remove_diag, pool=pool1, dropout=dropout, weight_sharing=weight_sharing, equi=equi) for i in range(num_blocks)])
        self.dec = self._make_decoder(latent_size, hidden_size, output_size, decoder_layers)
        self.remove_diag = remove_diag
        self.equi=equi

    def _make_decoder(self, latent_size, hidden_size, output_size, n_layers):
        if n_layers == 0:
            return nn.Linear(2*latent_size, output_size)
        else:
            hidden_layers = []
            for _ in range(n_layers-1): 
                hidden_layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            return nn.Sequential(
                nn.Linear(2*latent_size, hidden_size),
                nn.ReLU(),
                *hidden_layers,
                nn.Linear(hidden_size, output_size)
            )

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



class ImageEncoderWrapper(nn.Module):
    def __init__(self, encoder, output_size):
        super().__init__()
        self.encoder = encoder
        self.output_size = output_size

    def forward(self, inputs):
        encoded_batch = self.encoder(inputs.view(-1, *inputs.size()[-3:]))
        return encoded_batch.view(*inputs.size()[:-3], encoded_batch.size(-1))

class BertEncoderWrapper(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.output_size = bert.config.hidden_size

    def forward(self, inputs):
        ss, n_seqs, bert_inputs = inputs['set_size'], inputs['n_seqs'], inputs['inputs']
        encoded_seqs = self.bert(**bert_inputs).last_hidden_state
        if n_seqs == 1:
            out = encoded_seqs[:,0].reshape(-1, ss, encoded_seqs.size(-1))
        else:
            out = encoded_seqs[:,0].reshape(-1, ss, n_seqs, encoded_seqs.size(-1)).mean(2)
        return out

class EmbeddingEncoderWrapper(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.output_size=embed_dim

    def forward(self, inputs):
        return inputs


class MultiSetModel(nn.Module):
    def __init__(self, set_model, X_encoder, Y_encoder):
        super().__init__()
        self.set_model = set_model
        self.X_encoder = X_encoder
        self.Y_encoder = Y_encoder
        self.latent_size = set_model.input_size
        self.X_proj = nn.Linear(X_encoder.output_size, self.latent_size) if X_encoder.output_size != self.latent_size else None
        self.Y_proj = nn.Linear(Y_encoder.output_size, self.latent_size) if Y_encoder.output_size != self.latent_size else None

    def forward(self, X, Y, **kwargs):
        ZX = self.X_encoder(X)
        ZY = self.Y_encoder(Y)

        if self.X_proj is not None:
            ZX = self.X_proj(ZX)
        if self.Y_proj is not None:
            ZY = self.Y_proj(ZY)
        
        return self.set_model(ZX, ZY, **kwargs)



class CocoTrivialModel(nn.Module):
    def __init__(self, text_enc, img_enc, latent_size, hidden_size, output_size):
        self.text_encoder = text_enc
        self.img_encoder = img_enc
        self.decoder = nn.Sequential(
            nn.Linear(2*latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.X_proj = nn.Linear(X_encoder.output_size, self.latent_size) if X_encoder.output_size != self.latent_size else None
        self.Y_proj = nn.Linear(Y_encoder.output_size, self.latent_size) if Y_encoder.output_size != self.latent_size else None
    
    def forward(self, imgs, texts):
        ZX = self.img_encoder(imgs)
        ZY = self.text_encoder(texts)

        if self.X_proj is not None:
            ZX = self.X_proj(ZX)
        if self.Y_proj is not None:
            ZY = self.Y_proj(ZY)
        
        return self.set_model(ZX, ZY, **kwargs)


#
#   GAN Stuff
#


class SetDecoderBlock(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_heads, attn_size=None, ln=False, equi=False, nn_attn=False, dropout=0.1):
        super().__init__()
        attn_size = attn_size if attn_size is not None else input_size
        self.attn1 = MHA(input_size, attn_size, latent_size, num_heads, equi=equi, nn_attn=nn_attn)
        self.attn1 = MHA(input_size, attn_size, latent_size, num_heads, equi=equi, nn_attn=nn_attn)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, latent_size))
        if ln:
            self.ln0 = nn.LayerNorm(latent_size)
            self.ln1 = nn.LayerNorm(latent_size)
            self.ln2 = nn.LayerNorm(latent_size)

    def forward(self, Q, K, **kwargs):
        X = Q + self.attn(Q, Q, **kwargs)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln0', None) is None else self.ln0(X)
        X = X + self.attn(X, K, **kwargs)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln1', None) is None else self.ln1(X)
        X = X + self.fc(X)
        X = X if getattr(self, 'dropout', None) is None else self.dropout(X)
        X = X if getattr(self, 'ln2', None) is None else self.ln2(X)
        return X