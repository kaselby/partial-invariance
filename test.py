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
from train import evaluate
use_cuda=torch.cuda.is_available()

RUN_DIR="final-runs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default='final-runs')
    parser.add_argument('--target', type=str, default='wasserstein')
    parser.add_argument('--set_size', type=int, nargs=2, default=[100, 150])
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--steps', type=int, default=500)

    return parser.parse_args()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders



if __name__ == '__main__':
    args = parse_args()

    base_sample_kwargs={'n':args.n, 'set_size':args.set_size}
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
        generators = {'gmm':GaussianGenerator(num_outputs=2, return_params=exact_loss), 'nf':NFGenerator(32, 3, num_outputs=2, return_params=exact_loss)}
        #generators = {'gmm':PairedGaussianGenerator(return_params=exact_loss), 'nf':NFGenerator(32, 3, num_outputs=2, return_params=exact_loss)}
        data_kwargs={'gmm':{'nu':5, 'mu0':0, 's0':0.3}, 'nf':{}}
    else:
        generators={'corr':CorrelatedGaussianGenerator(return_params=exact_loss)}
        data_kwargs={'corr':{}}

    basedir=os.path.join(args.basedir, args.target)
    run_paths = glob.glob(os.path.join(basedir, args.run_name+"*"))
    results={}
    for name,generator in generators.items():
        sample_kwargs = {**base_sample_kwargs, **data_kwargs[name]}
        print("%s:"%name)
        seed = torch.randint(100, (1,)).item()
        for run_path in run_paths:
            run_name = run_path.split("/")[-1]
            all_runs = get_runs(run_path)
            if run_name not in results:
                results[run_name] = {}
            if len(all_runs) > 0:
                results[run_name][name]={'all_losses':[]}
                avg_loss=0
                for run_num in all_runs:
                    model = torch.load(os.path.join(run_path, run_num, "model.pt"))
                    model_loss_i = evaluate(model, generator, label_fct, 
                        sample_kwargs=sample_kwargs, steps=args.steps, criterion=nn.L1Loss(), normalize=normalize, exact_loss=exact_loss, seed=seed)
                    results[run_name][name]['all_losses'].append(model_loss_i)
                    avg_loss += model_loss_i
                    print("%s-%s Loss: %f" % (run_name, run_num, model_loss_i))
                avg_loss /= len(all_runs)
                print("%s Avg Loss: %f" % (run_name, avg_loss))
                results[run_name][name]['avg_loss'] = avg_loss
            else:
                model = torch.load(os.path.join(run_path, "model.pt"))
                model_loss = evaluate(model, generator, label_fct, 
                    sample_kwargs=sample_kwargs, steps=args.steps, criterion=nn.L1Loss(), normalize=normalize, exact_loss=exact_loss, seed=seed)
                print("%s Loss: %f" % (run_name, model_loss))
                results[run_name][name] = model_loss
        for baseline_name, baseline_fct in baselines.items():
            if baseline_name not in results:
                results[baseline_name] = {}
            baseline_loss = evaluate(baseline_fct, generator, label_fct, 
                sample_kwargs=sample_kwargs, steps=args.steps, criterion=nn.L1Loss(), normalize=False, exact_loss=exact_loss, seed=seed)
            print("%s Loss: %f" % (baseline_name, baseline_loss))
            results[baseline_name][name] = baseline_loss
    

    outfile = os.path.join("eval", args.run_name + "_results.txt")
    with open(outfile, 'w') as writer:
        for name in generators.keys():
            writer.write(name+":\n")
            for baseline_name in baselines.keys():
                writer.write("\t%s:%f" % (baseline_name, results[baseline_name][name]))
            for run_path in run_paths:
                run_name = run_path.split("/")[-1]
                if type(results[run_name][name]) is dict:
                    all_losses, avg_loss = results[run_name][name]['all_losses'], results[run_name][name]['avg_loss']
                    writer.write("\t%s:\n" % run_name)
                    writer.write("\t\tAll Losses:" + str(all_losses) + "\n")
                    writer.write("\t\tAvg Loss: %f\n"% avg_loss)
                else:
                    writer.write("\t%s:%f" % (run_name, results[run_name][name]))




