import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, RandomSampler, BatchSampler

import numpy as np
import os
import glob
import tqdm
import argparse

from icr import ICRDict
from models import MultiSetTransformer1

def load_vocab(filename):
    with open(filename, 'r') as reader:
        lines = reader.readlines()
    return [line.strip().lower() for line in lines]

def load_dataset_vecs(dataset, vec_dir, vocab_dir):
    voc = load_vocab(os.path.join(vocab_dir, dataset+".txt"))
    words_found=set()
    vec_files = glob.glob(os.path.join(vec_dir, "vecs*.pt"))
    dataset_vecs = {}
    for file in vec_files:
        vecs_i = torch.load(file)
        for word in voc:
            if not (word in words_found) and word in vecs_i:
                dataset_vecs[word] = vecs_i[word]
                words_found.add(word)
        del vecs_i
    words_left = len(voc) - len(words_found)
    print("Loaded %s. %d words missing." % (dataset, words_left))
    return dataset_vecs

class HyponomyDataset(Dataset):
    def __init__(self, dataset_name, data_dir, vec_dir, voc_dir, inverted_pairs=False, pca_dim=-1, max_vecs=-1, min_threshold=10):
        load_dict = load_dataset_vecs(dataset_name, vec_dir, voc_dir)
        self.vecs = ICRDict.from_dict(load_dict)
        dataset_path = os.path.join(data_dir, dataset_name + ".all")
        self.relations, self.pairs, self.labels = self.read_dataset(dataset_path, min_threshold, inverted_pairs=inverted_pairs)
        self.pca_dim=pca_dim
        self.max_vecs=max_vecs

    def _read_dataset(self, dataset_path, min_threshold, inverted_pairs=False):
        """Reads the hypernymy pairs, relation type and the true label from the given file and returns these
            four properties a separate lists.

        Parameters
        __________
        dataset_path: string
            Path of the dataset file. The file should contain one positive/negative pair per line. The format of each
            line should be of the following form:
                hyponym  hypernym    label   relation-type
            each separated by a tab.

        inverted_pairs: bool
            Whether only the positive pairs + all positive pairs inverted (switch hyponym <-> hypernym in positive
            pairs) should be returned. This can be helpful to check how well a model can the directionality of the
            hypernymy relation.

        Returns
        _______
        tuple:
            relations: np.array, pairs: list[(hyponym, hypernym)], labels: np.array(dtype=bool)
        """
        with open(dataset_path) as f:
            dataset = [tuple(line.strip().split("\t")) for line in f]

            for i in range(len(dataset)):
                if len(dataset[i]) < 4:
                    raise ValueError('Encountered invalid line in "%s" on line %d: %s' % (dataset_path, i, dataset[i]))

            w1, w2, labels, relations = zip(*dataset)
            pairs = list(zip(w1, w2))
            labels = (np.array(labels) == "True")

            if inverted_pairs:
                pos_pairs = [pairs[ix] for ix, lbl in enumerate(labels) if lbl]
                neg_pairs = [(p2, p1) for p1, p2 in pos_pairs]
                pairs = pos_pairs + neg_pairs
                labels = np.array([True] * len(pos_pairs) + [False] * len(neg_pairs))
                relations = ['hyper'] * len(pos_pairs) + ['inverted'] * len(neg_pairs)
        
        relations, pairs, labels = self._trim_dataset(relations, pairs, labels, min_threshold)

        return np.array(relations), pairs, labels

    def _trim_dataset(self, relations, pairs, labels, min_threshold):
        rnew, pnew, lnew = [], [], []
        for i, (w1,w2) in enumerate(pairs):
            n1 = -1 if w1 not in self.vecs else self.vecs[w1].n
            n2 = -1 if w2 not in self.vecs else self.vecs[w2].n
            if n1 >= min_threshold and n2 >= min_threshold:
                rnew.append(relations[i])
                pnew.append(pairs[i])
                lnew.append(labels[i])
        return rnew, pnew, lnew
                

    def __getitem__(self, index):
        w1,w2 = self.pairs[index]
        transform = self.vecs.pca(w1,w2, n_components=self.pca_dim)
        return (
            self.vecs[w1].get_vecs(transform=transform, max_vecs=self.max_vecs), 
            self.vecs[w2].get_vecs(transform=transform, max_vecs=self.max_vecs)
        ), self.labels[index]

    def __len__(self):
        return len(self.pairs)

def compare(w1, w2, vec_dicts, distance):
    vecs1 = vec_dicts[w1].cuda()
    vecs2 = vec_dicts[w2].cuda()
    return distance(vecs1, vecs2)


def train(model, dataset, steps, batch_size=64, lr=1e-3, save_every=5000, log_every=500, checkpoint_dir=None, output_dir=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fct = nn.BCELoss()
    out = nn.Sigmoid()

    current_step=0
    losses = []

    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            load_dict = torch.load(checkpoint_path)
            model.load_state_dict(load_dict['model'])
            optimizer.load_state_dict(load_dict['optimizer'])
            current_step = load_dict['step']
            losses = load_dict['losses']

    while current_step < steps:
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset, replacement=True), drop_last=True)
        for data, label in data_loader:
            optimizer.zero_grad()

            score = model(data)
            loss = loss_fct(out(score), label)
            loss.backward()
            optimizer.step()

            if (current_step + batch_size) // save_every > current_step // save_every:
                if checkpoint_dir is not None:
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'step':current_step, 'losses': losses}, checkpoint_path)

            if (current_step + batch_size) // log_every > current_step // log_every:
                losses.append(loss.item())

            current_step += batch_size

    if output_dir is not None:
        torch.save(model, os.path.join(output_dir, "model.pt"))
        torch.save({'loss':losses}, os.path.join(output_dir, "logs.pt"))

    return losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--voc_dir', type=str, default='./ICR/voc')
    parser.add_argument('--vec_dir', type=str, default='./ICR/vecs')
    parser.add_argument('--data_dir', type=str, default='./ICR/data')
    parser.add_argument('--pca_dim', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--max_vecs', type=int, default=250)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=5e-4)
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--output_dir', type=str, default="runs/hypeval")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = HyponomyDataset('HypNet_train', args.data_dir, args.vec_dir, args.voc_dir, pca_dim=args.pca_dim, max_vecs=args.max_vecs)
    model = MultiSetTransformer1(args.pca_dim, 1, 1, args.hidden_size, num_heads=args.n_heads, num_blocks=args.n_blocks, ln=True)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    output_dir = os.path.join(args.output_dir, args.run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(model, dataset, args.steps, batch_size=args.batch_size, lr=args.lr, checkpoint_dir=checkpoint_dir, output_dir=output_dir)




    