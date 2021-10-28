from numpy.core.arrayprint import _none_or_positive_arg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, RandomSampler, BatchSampler

import numpy as np
import os
import glob
import tqdm
import argparse
from sklearn.metrics import average_precision_score

from icr import ICRDict
from models import MultiSetTransformer1
from models2 import MultiSetTransformer

use_cuda = torch.cuda.is_available()

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

def generate_masks(X_lengths, Y_lengths):
    X_max, Y_max = max(X_lengths), max(Y_lengths)

    X_mask = torch.arange(X_max)[None, :] < torch.Tensor(X_lengths)[:, None]
    Y_mask = torch.arange(Y_max)[None, :] < torch.Tensor(Y_lengths)[:, None]

    mask_xx = X_mask.long()[:,:,None].matmul(X_mask.long()[:,:,None].transpose(1,2))
    mask_yy = Y_mask.long()[:,:,None].matmul(Y_mask.long()[:,:,None].transpose(1,2))
    mask_xy = X_mask.long()[:,:,None].matmul(Y_mask.long()[:,:,None].transpose(1,2))
    mask_yx = Y_mask.long()[:,:,None].matmul(X_mask.long()[:,:,None].transpose(1,2))

    return mask_xx, mask_xy, mask_yx, mask_yy


def pad_batch(inputs):
    d = inputs[0].shape[-1]
    lens = [x.shape[0] for x in inputs]
    maxlen = max(lens)
    batch = torch.zeros(len(inputs), maxlen, d)
    for i, elem in enumerate(inputs):
        batch[i, :lens[i], :] = torch.as_tensor(elem)
    return batch, lens

def collate_batch_with_padding(inputs):
    inputs_x, inputs_y, labels = zip(*inputs)
    batch_x, lens_x = pad_batch(inputs_x)
    batch_y, lens_y = pad_batch(inputs_y)
    labels = torch.stack([torch.as_tensor(x) for x in labels], dim=0)
    masks = generate_masks(lens_x, lens_y)

    return (batch_x, batch_y), masks, labels




class HyponomyDataset(Dataset):
    @staticmethod
    def _read_dataset(dataset_path, inverted_pairs=False):
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

        return np.array(relations), pairs, labels

    @staticmethod
    def _trim_dataset(vecs, pairs, min_threshold):
        valid_indices = []
        for i, (w1,w2) in enumerate(pairs):
            n1 = -1 if w1 not in vecs else vecs[w1].n
            n2 = -1 if w2 not in vecs else vecs[w2].n
            if n1 >= min_threshold and n2 >= min_threshold:
                valid_indices.append(i)
        return valid_indices

    @classmethod
    def from_file(cls, dataset_name, data_dir, vec_dir, voc_dir, inverted_pairs=False, min_threshold=10, pca_dim=-1, max_vecs=-1):
        load_dict = load_dataset_vecs(dataset_name, vec_dir, voc_dir)
        vecs = ICRDict.from_dict(load_dict)
        dataset_path = os.path.join(data_dir, dataset_name + ".all")
        relations, pairs, labels = cls._read_dataset(dataset_path, inverted_pairs=inverted_pairs)
        n0 = len(pairs)
        valid_indices=None
        if min_threshold >= 0:
            valid_indices = cls._trim_dataset(vecs, pairs, min_threshold)
            print("Dataset contains %d pairs. %d pairs removed after filtering." % (n0, n0-len(valid_indices)))
        return cls(vecs, relations, pairs, labels, pca_dim=pca_dim, max_vecs=max_vecs, valid_indices=valid_indices)

    def __init__(self, vecs, relations, pairs, labels, pca_dim=-1, max_vecs=-1, valid_indices=None):
        assert len(pairs) == len(labels)
        self.vecs = vecs
        self.relations=relations
        self.pairs=pairs
        self.labels = labels
        self.pca_dim=pca_dim
        self.max_vecs=max_vecs
        if valid_indices is None:
            self.n = len(pairs)
            self.valid_indices = list(range(self.n))
        else:
            self.n = len(valid_indices)
            self.valid_indices = valid_indices
        self.positive = [i for i in self.valid_indices if labels[i] == True]
        self.negative = [i for i in self.valid_indices if labels[i] == False]

    def split(self, frac):
        n_split = int(frac * self.n)
        if self.valid_indices is None:
            inds1, inds2 = range(self.n)[:n_split], range(self.n)[n_split:]
        else:
            inds1, inds2 = self.valid_indices[:n_split], self.valid_indices[n_split:]
        r1, r2 = [self.relations[i] for i in inds1], [self.relations[i] for i in inds2]
        p1, p2 = [self.pairs[i] for i in inds1], [self.pairs[i] for i in inds2]
        l1, l2 = [self.labels[i] for i in inds1], [self.labels[i] for i in inds2]
        d1 = HyponomyDataset(self.vecs, r1, p1, l1, self.pca_dim, self.max_vecs)
        d2 = HyponomyDataset(self.vecs, r2, p2, l2, self.pca_dim, self.max_vecs)
        return d1, d2

    def __getitem__(self, index):
        j = index if self.valid_indices is None else self.valid_indices[index]
        w1,w2 = self.pairs[j]
        transform = self.vecs.pca(w1,w2, n_components=self.pca_dim).transform if self.pca_dim > 0 else None
        return self.vecs[w1].get_vecs(transform=transform, max_vecs=self.max_vecs), \
            self.vecs[w2].get_vecs(transform=transform, max_vecs=self.max_vecs), \
            self.labels[index]

    def __len__(self):
        return self.n

    def get_invalid(self):
        assert self.valid_indices is not None
        invalid_indices = [i for i in range(len(self.pairs)) if not i in self.valid_indices]
        return [self.pairs[i] for i in invalid_indices], [self.labels[i] for i in invalid_indices]

    def get_classes(self):
        return self.negative, self.positive


class RandomSubsampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, dataset, idx_by_class, generator=None) -> None:
        self.idx_by_class = idx_by_class
        self.n_classes = len(self.idx_by_class)
        self.n = len(dataset)
        self.generator = generator

    def __iter__(self):
        samplers_by_class = [iter(RandomSampler(self.idx_by_class[i], replacement=True, num_samples=self.n)) for i in range(self.n_classes)]
        for j_i in torch.randint(self.n_classes, size=(self.n,)):
            t_j = next(samplers_by_class[j_i])
            yield t_j

    def __len__(self) -> int:
        return self.n


def compare(w1, w2, vec_dicts, distance):
    vecs1 = vec_dicts[w1].cuda()
    vecs2 = vec_dicts[w2].cuda()
    return distance(vecs1, vecs2)




def train(model, dataset, steps, eval_dataset=None, batch_size=64, lr=1e-3, save_every=5000, log_every=10, checkpoint_dir=None, output_dir=None, reweight=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fct = nn.BCEWithLogitsLoss()

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
        if reweight:
            sampler = RandomSubsampler(dataset, dataset.get_classes())   
        else:
            sampler = RandomSampler(dataset, replacement=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_batch_with_padding, drop_last=True)
        for data, masks, labels in tqdm.tqdm(data_loader):
            optimizer.zero_grad()

            if use_cuda:
                data = [X.cuda() for X in data]
                masks = [mask.cuda() for mask in masks]
                labels = labels.cuda()

            score = model(*data, masks=masks)
            loss = loss_fct(score.squeeze(-1), labels.float())

            if score.isnan().any() or loss.isnan().any():
                print("nan1")
                import pdb;pdb.set_trace()

            loss.backward()
            optimizer.step()

            if any([x.isnan().any().item() for x in model.parameters()]):
                print("nan2")
                import pdb;pdb.set_trace()

            if (current_step + batch_size) // save_every > current_step // save_every:
                if checkpoint_dir is not None:
                    if os.path.exists(checkpoint_path):
                        os.remove(checkpoint_path)
                    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'step':current_step, 'losses': losses}, checkpoint_path)

            if (current_step + batch_size) // log_every > current_step // log_every:
                losses.append(loss.item())

            current_step += batch_size

    logs = {'losses':losses}
    if eval_dataset is not None:
        acc, prec = evaluate(model, eval_dataset, batch_size=batch_size, append_missing=False)
        logs['eval_acc'] = acc
        logs['eval_prec'] = prec

    if output_dir is not None:
        torch.save(model, os.path.join(output_dir, "model.pt"))
        torch.save(logs, os.path.join(output_dir, "logs.pt"))

    return losses

def evaluate(model, dataset, batch_size=64, append_missing=False):
    all_logits = torch.zeros(len(dataset))
    all_labels = torch.zeros(len(dataset))

    data_loader=DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_with_padding)
    for i, (data, masks, labels) in enumerate(data_loader):
        j_min = i * batch_size
        j_max = min(len(dataset), (i + 1) * batch_size)

        if use_cuda:
            data = [X.cuda() for X in data]
            masks = [mask.cuda() for mask in masks]

        out = model(*data, masks=masks)

        all_logits[j_min:j_max] = out.squeeze(-1).cpu().detach()
        all_labels[j_min:j_max] = labels.detach()
    
    if dataset.valid_indices is not None and append_missing:
        _, missing_labels = dataset.get_invalid()
        n_missing = len(missing_labels)
        all_logits = torch.cat([all_logits, torch.zeros(n_missing)], dim=0)
        all_labels = torch.cat([all_labels, torch.Tensor(missing_labels)], dim=0)
    
    def get_accuracy(labels, logits):
        return ((labels*2 - 1) * logits > 0).float().sum() / logits.size(0)

    accuracy = get_accuracy(all_labels, all_logits)
    precision = average_precision_score(all_labels.numpy(), all_logits.numpy())

    return accuracy, precision
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--voc_dir', type=str, default='./ICR/voc')
    parser.add_argument('--vec_dir', type=str, default='./ICR/vecs/hypeval')
    parser.add_argument('--data_dir', type=str, default='./ICR/data')
    parser.add_argument('--pca_dim', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--max_vecs', type=int, default=250)
    parser.add_argument('--steps', type=int, default=150000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="runs/hypeval")
    parser.add_argument('--reweight', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = HyponomyDataset.from_file('HypNet_train', args.data_dir, args.vec_dir, args.voc_dir, 
        pca_dim=args.pca_dim, min_threshold=args.pca_dim, max_vecs=args.max_vecs)
    train_dataset, eval_dataset = dataset.split(0.85)
    model = MultiSetTransformer(args.pca_dim, args.latent_size, args.hidden_size, 1, num_heads=args.n_heads, num_blocks=args.n_blocks, ln=True)

    if use_cuda:
        model = model.cuda()

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    output_dir = os.path.join(args.output_dir, args.run_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(model, train_dataset, args.steps, eval_dataset=eval_dataset, batch_size=args.batch_size, lr=args.lr, 
        checkpoint_dir=checkpoint_dir, output_dir=output_dir, reweight=args.reweight)




    