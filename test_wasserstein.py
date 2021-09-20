import io
import os
import argparse
import random

import torch
import fasttext
import numpy as np

from utils import show_examples, wasserstein, generate_gaussian_mixture, generate_multi

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--vec_path1', type=str, default='cc.en.32.bin')
    parser.add_argument('--vec_path2', type=str, default='cc.fr.32.bin')
    parser.add_argument('--normalize', action='store_true')

    return parser.parse_args()

def sample_vecs(ft1, ft2, scale=-1):
    def get_samples(ft, bs, set_size=(100,150)):
        n_samples=random.randint(*set_size)
        vecs = []
        for i in range(bs):
            words = random.sample(ft.get_words(), n_samples)
            vecs_i = [ft[x].tolist() for x in words]
            vecs.append(vecs_i)
        out = torch.Tensor(vecs)
        if scale > 0:
            out /= scale
        return out
    return lambda n: (get_samples(ft1, n), get_samples(ft2, n))


if __name__ == '__main__':
    args = parse_args()
    print("test")

    model = torch.load(os.path.join("runs", args.run_name, "model.pt"))

    ft = fasttext.load_model("cc.en.32.bin")
    if args.normalize:
        scale = np.linalg.norm(ft.get_input_matrix(), axis=1).mean()
    else:
        scale=-1

    show_examples(model, generate_multi(generate_gaussian_mixture), wasserstein, n=32)
    show_examples(model, sample_vecs(ft, scale=scale), wasserstein)
    




