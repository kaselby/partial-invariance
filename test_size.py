import io
import os
import argparse
import random

import torch
import torch.nn as nn
import numpy as np
import tqdm

from utils import *
from train import evaluate
use_cuda=torch.cuda.is_available()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, nargs='+')
    parser.add_argument('--target', type=str, default='wasserstein')

    return parser.parse_args()

def get_runs(run_name, basedir="runs"):
    subfolders = [f.name for f in os.scandir(os.path.join(basedir, run_name)) if f.is_dir()]
    return subfolders

def eval_all(sizes, sample_kwargs, *args, **kwargs):
    losses = torch.zeros_like(sizes)
    for i in range(sizes.size(0)):
        sample_kwargs['set_size']=(sizes[i],sizes[i]+1)
        losses[i] = evaluate(*args, sample_kwargs=sample_kwargs, **kwargs)
    return losses



if __name__ == '__main__':
    args = parse_args()
    
    sizes = torch.linspace(2,9,20).exp().round()

    sample_kwargs={'n':2}
    if args.target == 'wasserstein':
        baselines = {'sinkhorn_default':wasserstein, 'sinkhorn_exact': lambda X,Y: wasserstein(X,Y, blur=0.001,scaling=0.98)}
        label_fct=wasserstein_exact
        exact_loss=False
        normalize='scale'
    elif args.target == 'kl':
        baselines = {'knn':kl_knn}
        label_fct=kl_mc
        exact_loss=True
        normalize='whiten'
    elif args.target == 'mi':
        baselines={'kraskov':kraskov_mi1}
        label_fct=mi_corr_gaussian
        exact_loss=True
        normalize='none'
    else:
        raise NotImplementedError()

    if args.target != 'mi':
        generator = GaussianGenerator(num_outputs=2, return_params=exact_loss)
    else:
        generator = CorrelatedGaussianGenerator(return_params=exact_loss)

    seed = torch.randint(100, (1,)).item()
    run_name = args.run_name
    all_runs = get_runs(run_name, "runs")
    results={}
    if len(all_runs) > 0:
        avg_losses = torch.zeros_like(sizes)
        for run_num in all_runs:
            model = torch.load(os.path.join("runs", run_name, run_num, "model.pt"))
            avg_losses = avg_losses + eval_all(sizes, sample_kwargs, model, generator, label_fct, steps=200, 
                criterion=nn.L1Loss(), normalize=normalize, exact_loss=exact_loss, seed=seed)
        results['model'] = avg_losses / len(all_runs)
    else:
        model = torch.load(os.path.join("runs", run_name, "model.pt"))
        model_losses = eval_all(sizes, sample_kwargs, model, generator, label_fct, 
            steps=500, criterion=nn.L1Loss(), normalize=normalize, exact_loss=exact_loss, seed=seed)
        results['model'] = model_losses
    for baseline_name, baseline_fct in baselines.items():
        baseline_losses = eval_all(sizes, sample_kwargs, baseline_fct, generator, label_fct, 
            steps=200, criterion=nn.L1Loss(), normalize=False, exact_loss=exact_loss, seed=seed)
        results[baseline_name] = baseline_losses

    torch.save(results, os.path.join("runs", run_name, "ss_losses.pt"))
