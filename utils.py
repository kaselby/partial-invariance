import torch

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

def poisson_loss(outputs, targets):
    return -1 * (targets * outputs - torch.exp(outputs)).mean()

def linear_block(input_size, hidden_size, output_size, num_layers, activation_fct=nn.ReLU):
    if num_layers == 0:
        layers = [nn.Linear(input_size, hidden_size), activation_fct(), nn.Linear(hidden_size, output_size)]
    elif num_layers > 0:
        layers = [nn.Linear(input_size, hidden_size), activation_fct()]
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
    else:
        raise AssertionError("num_layers must be >= 0")
    return nn.Sequential(*layers)