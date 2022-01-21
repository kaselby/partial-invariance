
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


from train_gan import eval_disc, summarize_eval

use_cuda=torch.cuda.is_available()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default='final-runs/meta-dataset/discriminator')
    parser.add_argument('--n_episodes', type=int, default=8)
    parser.add_argument('--set_size', type=int, nargs=2, default=[10, 30])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--episode_classes', type=int, default=200)
    parser.add_argument('--episode_datasets', type=int, default=11)
    parser.add_argument('--episode_length', type=int, default=80)
    parser.add_argument('--outfile', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_kwargs={
        'set_size':(10,30),
        'p_aligned': 0.5,
        'p_dataset': 0.3,
        'p_same': 0.5
    }

    test_generator = MetaDatasetGenerator(image_size=image_size, split=Split.TEST, device=device)
    

    model_dir = os.path.join(args.basedir, args.run_name)
    runs = get_runs(model_dir)
    accs = torch.zeros(len(runs))
    for i, run_num in enumerate(runs):
        model = torch.load(os.path.join(model_dir, run_num, 'model.pt'))
        acc_i =0
        for j in range(args.n_episodes):
            episode = test_generator.get_episode(args.episode_classes, args.episode_datasets)
            acc = eval_disc(model, episode, args.episode_length, args.batch_size, data_kwargs)
            acc_i += acc
        accs[i] = acc_i / args.n_episodes
    avg_acc = accs.mean()
    std = accs.std()
    
    with open(args.outfile, 'a') as writer:
        writer.write("%s: \tAvg:%f\tStdev:%f\tAccs:%s" % (args.run_name, avg_acc.item(), std.item(), str(accs.tolist())))

                        



