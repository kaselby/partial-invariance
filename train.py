from models import *
from utils import generate_gaussian_mixture, wasserstein, generate_gaussian_variable_dim_multi, generate_gaussian_mixture_variable_dim_multi, generate_multi
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import os
import shutil
import glob
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--scaleinv', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--scaling', default=0.5)
    parser.add_argument('--blur', default=0.05)
    parser.add_argument('--equi', action='store_true')

    return parser.parse_args()

def train(model, sample_fct, label_fct, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, lr=1e-5, checkpoint_dir=None, output_dir=None, save_every=1000, sample_kwargs={}, label_kwargs={}):
    #model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    initial_step=1
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                load_dict = torch.load(checkpoint_path)
                model, optimizer, initial_step, losses = load_dict['model'], load_dict['optimizer'], load_dict['step'], load_dict['losses']

    for i in tqdm.tqdm(range(initial_step,steps+1)):
        optimizer.zero_grad()
        if exact_loss:
            X, theta = sample_fct(batch_size, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
                #theta = [t.cuda() for t in theta]
            labels = label_fct(*theta, **label_kwargs).squeeze(-1)
        else:
            X = sample_fct(batch_size, **sample_kwargs)
            if use_cuda:
                X = [x.cuda() for x in X]
            labels = label_fct(*X, **label_kwargs)
        loss = criterion(model(*X).squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % save_every == 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':losses}, checkpoint_path)
    
    if output_dir is not None:
        torch.save(model._modules['module'], os.path.join(output_dir,"model.pt"))  
        torch.save({'losses':losses}, os.path.join(output_dir,"logs.pt"))   

    return losses

if __name__ == '__main__':
    args = parse_args()
    run_dir = os.path.join("runs", args.run_name)
    '''if os.path.exists(run_dir):
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise Exception("Folder exists and overwrite is set to false.")'''
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")

    if args.equi:
        model=EquiMultiSetTransformer1(1,1, dim_hidden=16, ln=True, remove_diag=True, num_blocks=2).to(device)
    else:
        DIM=32
        model=MultiSetTransformer1(DIM, 1,1, dim_hidden=256, ln=True, remove_diag=True, num_blocks=2).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    sample_kwargs={'set_size':(10,150)}
    label_kwargs={'scaling':args.scaling, 'blur':args.blur}
    if args.equi:
        sample_kwargs['dims'] = (24,40)
        sample_kwargs['normalize'] = args.normalize
        sample_kwargs['scaleinv'] = args.scaleinv
        sample_fct = generate_gaussian_mixture_variable_dim_multi
    else:
        sample_kwargs['n'] = DIM
        sample_fct = generate_multi(generate_gaussian_mixture, scaleinv=args.scaleinv, normalize=args.normalize)
    losses = train(model, sample_fct, wasserstein, checkpoint_dir=os.path.join(args.checkpoint_dir, args.run_name), \
        output_dir=run_dir, criterion=nn.MSELoss(), steps=40000, lr=5e-4, batch_size=128, \
        sample_kwargs=sample_kwargs, label_kwargs=label_kwargs)


'''
d=2
hs=32
nh=4
ln=True
n_blocks=2
model= EquiEncoder(hs, n_blocks, nh, ln).cuda()
losses=train(model, generate_gaussian_nd, wasserstein, criterion=nn.MSELoss(), steps=20000, lr=1e-3, n=2, set_size=(50,75))
'''


