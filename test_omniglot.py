import io
import os
import argparse
from posixpath import basename
import random

import torch
import torch.nn as nn
import fasttext
import numpy as np
import tqdm
import glob

from utils import *
from generators import *
from train_omniglot import *
use_cuda=torch.cuda.is_available()

RUN_DIR="final-runs"

def evaluate(model, eval_generator, steps, poisson=False, batch_size=64, data_kwargs={}):
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    n_correct = 0
    l1 = 0
    l2 = 0
    with torch.no_grad():
        for i in range(steps):
            (X,Y), target = eval_generator(batch_size, **data_kwargs)
            out = model(X,Y).squeeze(-1)
            if poisson:
                out = torch.exp(out)
            n_correct += torch.logical_or(torch.eq(out.ceil(), target.int()), torch.eq(out.ceil()-1, target.int())).sum().item()
            l1 += l1loss(out, target).mean()
            l2 += l2loss(out, target).mean()
    acc = n_correct / (batch_size * steps)
    l1 = l1 / steps
    l2 = l2 / steps
    return {'acc':acc, 'l1':l1.item(), 'l2':l2.item()}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default='final-runs2')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--set_size', type=int, nargs=2, default=[10,30])
    parser.add_argument('--data_dir', type=str, default='./data')

    return parser.parse_args()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders



if __name__ == '__main__':
    args = parse_args()

    data_kwargs = {'set_size':args.set_size}

    if args.dataset == "mnist":
        _, test_dataset = load_mnist(args.data_dir)
        generator_cls = ImageCooccurenceGenerator
    else:
        _, _, test_dataset = load_omniglot(args.data_dir)
        generator_cls = OmniglotCooccurenceGenerator
        data_kwargs['n_chars'] = 50
    
    device=torch.device("cuda")
    test_generator = generator_cls(test_dataset, device)



    basedir=os.path.join(args.basedir, args.dataset)
    run_paths = glob.glob(os.path.join(basedir, args.run_name+"*"))
    results={}
    for run_path in run_paths:
        if os.path.isdir(run_path):
            run_name = run_path.split("/")[-1]
            all_runs = get_runs(run_path)
            if run_name not in results:
                results[run_name] = {}
            if len(all_runs) > 0:
                results[run_name] = {}
                for run_num in all_runs:
                    model = torch.load(os.path.join(run_path, run_num, "model.pt"))
                    model_losses_i = evaluate(model, test_generator, args.steps, data_kwargs=data_kwargs)
                    for k,v in model_losses_i.items():
                        if k not in results[run_name]:
                            results[run_name][k] = []
                        results[run_name][k].append(v)
                    print(run_name, ": ", model_losses_i)
    

    outfile = os.path.join("eval", args.run_name + "_results.txt")
    with open(outfile, 'w') as writer:
        for run_path in run_paths:
            run_name = run_path.split("/")[-1]
            for k, v in results[run_name].items():
                writer.write(k + ":" + str(v) + "\n")




