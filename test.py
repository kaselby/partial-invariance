import io
import os
import argparse
import random

import torch
import torch.nn as nn
import fasttext
import numpy as np
import tqdm

from utils import *
from train import evaluate
use_cuda=torch.cuda.is_available()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--target', type=str, default='wasserstein')
    parser.add_argument('--normalize', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("test")

    model = torch.load(os.path.join("runs", args.run_name, "model.pt"))

    sample_kwargs={'n':32, 'set_size':(10,150)}
    if args.target == 'wasserstein':
        baselines = {'sinkhorn_default':wasserstein, 'sinkhorn_exact': lambda X,Y: wasserstein(X,Y, blur=0.001,scaling=0.98)}
        label_fct=wasserstein_exact
        exact_loss=False
    elif args.target == 'kl':
        baselines = {'knn':kl_knn}
        label_fct=kl_mc
        exact_loss=True
    else:
        raise NotImplementedError()
    generators = {'gmm':GaussianGenerator(num_outputs=2, return_params=exact_loss), 'nf':NFGenerator(32, 3, num_outputs=2, return_params=exact_loss)}

    for name,generator in generators.items():
        model_loss, baseline_losses = evaluate(model, baselines, generator, label_fct, 
            sample_kwargs=sample_kwargs, steps=1000, criterion=nn.L1Loss(), normalize=args.normalize, exact_loss=exact_loss)

        print("%s:"%name)
        print("Model Loss:", model_loss)
        for baseline_name, baseline_loss in baseline_losses.items():
            print("%s Losses:" % baseline_name, baseline_loss)
