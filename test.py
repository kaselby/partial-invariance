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
use_cuda=torch.cuda.is_available()

def evaluate(model, baselines, generators, label_fct, exact_loss=False, batch_size=64, sample_kwargs={}, label_kwargs={}, criterion=nn.L1Loss(), steps=5000):
    model_losses = []
    baseline_losses = {k:[] for k in baselines.keys()}
    with torch.no_grad():
        for generator in generators:
            for i in tqdm.tqdm(range(steps)):
                if exact_loss:
                    X, theta = generator(batch_size, **sample_kwargs)
                    if use_cuda:
                        X = [x.cuda() for x in X]
                        #theta = [t.cuda() for t in theta]
                    labels = label_fct(*theta, **label_kwargs).squeeze(-1)
                else:
                    X = generator(batch_size, **sample_kwargs)
                    if use_cuda:
                        X = [x.cuda() for x in X]
                    labels = label_fct(*X, **label_kwargs)
                model_loss = criterion(model(*X).squeeze(-1), labels)
                model_losses.append(model_loss.item())
                for baseline_name, baseline_fct in baselines.items():
                    baseline_loss = criterion(baseline_fct(*X), labels)
                    baseline_losses[baseline_name].append(baseline_loss.item())
    return sum(model_losses)/len(model_losses), {k:sum(v)/len(v) for k,v in baseline_losses.items()}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--target', type=str, default='wasserstein')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("test")

    model = torch.load(os.path.join("runs", args.run_name, "model.pt"))

    sample_kwargs={'n':32, 'set_size':(10,150)}
    if args.target == 'wasserstein':
        baselines = {'sinkhorn_default':wasserstein, 'sinkhorn_exact': lambda X,Y: wasserstein(X,Y, blur=0.001,scaling=0.98)}
    elif args.target == 'kl':
        baselines = {'knn':kl_knn}
    else:
        raise NotImplementedError()
    generators = [GaussianGenerator(num_outputs=2, normalize=True), NFGenerator(32, 3, num_outputs=2, normalize=True)]
    model_loss, baseline_losses = evaluate(model, baselines, generators, wasserstein_exact, 
        sample_kwargs=sample_kwargs, steps=1000, criterion=nn.L1Loss())

    print("Model Loss:", model_loss)
    for baseline_name, baseline_loss in baseline_losses.items():
        print("%s Losses:" % baseline_name, baseline_loss)
