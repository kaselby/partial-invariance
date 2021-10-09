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

from icr import ICRDict, avg_nn_dist
from models import MultiSetTransformer1
from train_hypnet import HyponomyDataset, collate_batch_with_padding
from utils import kl_knn

use_cuda = torch.cuda.is_available()

def evaluate_model(model, dataset, batch_size=64):
    all_logits = torch.zeros(len(dataset))
    all_labels = torch.zeros(len(dataset))

    data_loader=DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_with_padding)
    with torch.no_grad():
        for i, (data, masks, labels) in tqdm.tqdm(enumerate(data_loader)):
            j_min = i * batch_size
            j_max = min(len(dataset), (i + 1) * batch_size)

            if use_cuda:
                data = [X.cuda() for X in data]
                masks = [mask.cuda() for mask in masks]

            out = model(*data, masks=masks)

            all_logits[j_min:j_max] = out.squeeze(-1).cpu().detach()
            all_labels[j_min:j_max] = labels.detach()

    def get_accuracy(labels, logits):
        return ((labels*2 - 1) * logits > 0).float().sum() / logits.size(0)

    accuracy = get_accuracy(all_labels, all_logits)
    precision = average_precision_score(all_labels.numpy(), all_logits.numpy())

    return accuracy, precision

def evaluate_fct(fct, dataset, batch_size=64):
    all_logits = torch.zeros(len(dataset))
    all_labels = torch.zeros(len(dataset))

    for i in tqdm.tqdm(range(len(dataset))):
        v1, v2, l = dataset[i]

        if use_cuda:
            v1, v2 = v1.cuda(), v2.cuda()

        all_logits[i] = fct(v1, v2)
        all_labels[i] = l
    
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
    parser.add_argument('--max_vecs', type=int, default=250)
    parser.add_argument('--output_dir', type=str, default="runs/hypeval")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = HyponomyDataset.from_file('HypNet_test', args.data_dir, args.vec_dir, args.voc_dir, pca_dim=args.pca_dim, max_vecs=args.max_vecs)

    model = torch.load(os.path.join("runs", args.run_name, 'model.pt'))
    baseline_fcts = {'avg_nn_dist': avg_nn_dist, 'kl':kl_knn}

    model_acc, model_prec = evaluate_model(model, dataset)
    print("Model Accuracy: %f" % model_acc)
    print("Model Precision: %f" % model_prec)

    for name, fct in baseline_fcts.items():
        baseline_acc, baseline_prec = evaluate_fct(fct, dataset)
        print("%s Accuracy: %f" % (name, baseline_acc))
        print("%s Precision: %f" % (name, baseline_prec))