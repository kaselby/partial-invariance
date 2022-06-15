from train_hypnet import HyponomyDataset, evaluate, collate_batch_with_padding
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

use_cuda = torch.cuda.is_available()

def read_dataset(dataset_path, inverted_pairs=False):
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

DATA_DIR="ICR/data"
datasets = glob.glob(os.path.join(DATA_DIR, "*.all"))
for dataset in datasets:
    print(dataset)
    r,p,l = HyponomyDataset._read_dataset(dataset, 1)
    print("Positive: %d" % (l==True).sum())
    print("Negative: %d" % (l==False).sum())

'''
dataset = HyponomyDataset('HypNet_test', "ICR/data", "ICR/vecs/hypeval2", "ICR/voc", pca_dim=10, max_vecs=250)

#model = torch.load("runs/hypeval/test_hypnet_/model.pt")
#logits,labels=evaluate(model,dataset)

model = MultiSetTransformer1(10, 1, 1, 128, num_heads=8, num_blocks=2, ln=True).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fct = nn.BCEWithLogitsLoss()
data_loader = DataLoader(dataset, batch_size=64, sampler=RandomSampler(dataset, replacement=True), collate_fn=collate_batch_with_padding, drop_last=True)

data,masks,labels = iter(data_loader).__next__()
optimizer.zero_grad()

data = [X.cuda() for X in data]
masks = [mask.cuda() for mask in masks]
labels = labels.cuda()

score = model(*data, masks=masks)
loss = loss_fct(score.squeeze(-1), labels.float())
#loss.backward()
#optimizer.step()

Q = mab_x.fc_q(Q)
K, V = mab_x.fc_k(K), mab_x.fc_v(K)

dim_split = mab_x.dim_V // mab_x.num_heads
Q_ = torch.stack(Q.split(dim_split, 2), 0)
K_ = torch.stack(K.split(dim_split, 2), 0)
V_ = torch.stack(V.split(dim_split, 2), 0)

E = Q_.matmul(K_.transpose(2,3))/math.sqrt(mab_x.dim_V)
A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3)
O = torch.cat((Q_ + A.matmul(V_)).split(1, 0), 3).squeeze(0)
O = O if getattr(mab_x, 'ln0', None) is None else mab_x.ln0(O)
O = O + F.relu(mab_x.fc_o(O))
O = O if getattr(mab_x, 'ln1', None) is None else mab_x.ln1(O)

'''

'''
    Q = mab_x.fc_q(Q)
    K, V = mab_x.fc_k(K), mab_x.fc_v(K)
    dim_split = mab_x.dim_V // mab_x.num_heads
    Q_ = torch.stack(Q.split(dim_split, 2), 0)
    K_ = torch.stack(K.split(dim_split, 2), 0)
    V_ = torch.stack(V.split(dim_split, 2), 0)
    E = Q_.matmul(K_.transpose(2,3))/math.sqrt(mab_x.dim_V)
    A = masked_softmax(E, mask.unsqueeze(0).expand_as(E), dim=3) if mask is not None else torch.softmax(E,3)
    O = torch.cat((Q_ + A.matmul(V_)).split(1, 0), 3).squeeze(0)
    O = O if getattr(mab_x, 'ln0', None) is None else mab_x.ln0(O)
    O = O + F.relu(mab_x.fc_o(O))
    O = O if getattr(mab_x, 'ln1', None) is None else mab_x.ln1(O)
    return O
'''