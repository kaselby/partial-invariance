import io
import os
import argparse
import random

import torch

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
    parser.add_argument('--vec_path', type=str, default='cc.en.32.bin')

    return parser.parse_args()

def sample_vecs(ft):
    def get_samples(bs, set_size=(100,150)):
        n_samples=random.randint(*set_size)
        vecs = []
        for i in range(bs):
            words = random.sample(ft.get_words(), n_samples)
            vecs_i = [ft[x].tolist() for x in words]
            vecs.append(vecs_i)
        return torch.Tensor(vecs)
    return lambda n: (get_samples(n), get_samples(n))


if __name__ == '__main__':
    args = parse_args()

    model = torch.load(os.path.join("runs", args.run_name, "model.pt"))




