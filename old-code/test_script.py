from models2 import *
from train import train, evaluate
from utils import *
from generators import *

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from train_gan import train_synth, eval_synth

n=8
bs=64
lr=1e-3
ls=8
hs=16
nb=4
dl=1
nh=4
ln=True
rd=False
equi=False
dropout=0
normalize='none'
steps=30000
set_size=(10,30)
warmup=-1
generator = DistinguishabilityGenerator(torch.device('cuda'))
data_kwargs={'set_size':set_size, 'n':n}
weight_sharing='none'
merge='lambda'

model1 = MultiSetTransformer(n, ls, hs, 1, num_blocks=4, num_heads=nh, decoder_layers=dl, ln=ln, remove_diag=rd, equi=equi, dropout=dropout, weight_sharing=weight_sharing, merge=merge).cuda()
opt = torch.optim.Adam(model1.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-8, total_iters=warmup) if warmup > 0 else None
model1, (train_losses, eval_accs, test_acc)=train_synth(model1, opt, generator, steps, scheduler=scheduler, batch_size=bs, data_kwargs=data_kwargs)
print(eval_accs,"\n",test_acc)