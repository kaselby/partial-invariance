from models import *
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import os
import shutil

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--n_gpus', type=int, default=1)
    return parser.parse_args()

def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main(rank, world_size, out_dir):
    setup(rank, world_size)

    model=EquiMultiSetTransformer1(1,1, dim_hidden=16, ln=True, remove_diag=True, num_blocks=2).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    losses = train(rank, world_size, ddp_model, generate_gaussian_variable_dim_multi, wasserstein, steps=30000, lr=5e-4, batch_size=int(64/world_size))

    if rank == 0:
        torch.save(model.module, os.path.join(out_dir,"model.pt"))  
        torch.save({'losses':losses}, os.path.join(out_dir,"logs.pt"))      


def train(rank, world_size, model, sample_fct, label_fct, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, lr=1e-5, *sample_args, **sample_kwargs):
    #model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(1,steps+1):
        optimizer.zero_grad()
        if exact_loss:
            X, theta = sample_fct(batch_size, *sample_args, **sample_kwargs)
            X = [x.to(rank) for x in X]
                #theta = [t.cuda() for t in theta]
            labels = label_fct(*theta).squeeze(-1)
        else:
            X = sample_fct(batch_size, *sample_args, **sample_kwargs)
            X = [x.to(rank) for x in X]
            labels = label_fct(*X)
        loss = criterion(model(*X).squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

if __name__ == '__main__':
    args = parse_args()
    run_dir = os.path.join("runs", args.run_name)
    if os.path.exists(run_dir):
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise Exception("Folder exists and overwrite is set to false.")

    os.makedirs(run_dir)

    mp.spawn(
        main,
        args=(args.n_gpus, run_dir),
        nprocs=args.n_gpus
    )

    

