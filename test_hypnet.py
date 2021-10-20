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
from train_hypnet import HyponomyDataset, collate_batch_with_padding, evaluate
from utils import avg_cross_nn_dist, kl_knn

use_cuda = torch.cuda.is_available()


def evaluate_fct(fct, dataset, append_missing=False):
    all_logits = torch.zeros(len(dataset))
    all_labels = torch.zeros(len(dataset))

    for i in tqdm.tqdm(range(len(dataset))):
        v1, v2, l = dataset[i]
        v1, v2, l = torch.Tensor(v1).unsqueeze(0), torch.Tensor(v2).unsqueeze(0), torch.BoolTensor([l])
        if use_cuda:
            v1, v2 = v1.cuda(), v2.cuda()

        all_logits[i] = fct(v1, v2).squeeze(0)
        all_labels[i] = l

    if dataset.valid_indices is not None and append_missing:
        _, missing_labels = dataset.get_invalid()
        n_missing = len(missing_labels)
        all_logits = torch.cat([all_logits, torch.zeros(n_missing)], dim=0)
        all_labels = torch.cat([all_logits, torch.Tensor(missing_labels)], dim=0)
    
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
    parser.add_argument('--output_dir', type=str, default="runs")
    parser.add_argument('--dataset', type=str, default="HypNet_test")
    parser.add_argument('--append_missing', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = HyponomyDataset.from_file(args.dataset, args.data_dir, args.vec_dir, args.voc_dir, pca_dim=args.pca_dim, max_vecs=args.max_vecs)

    model = torch.load(os.path.join("runs", args.run_name, 'model.pt'))
    baseline_fcts = {'avg_nn_dist': avg_cross_nn_dist, 'kl':kl_knn}

    model_acc, model_prec = evaluate(model, dataset, append_missing=True)
    print("Model Accuracy: %f" % model_acc)
    print("Model Precision: %f" % model_prec)

    for name, fct in baseline_fcts.items():
        baseline_acc, baseline_prec = evaluate_fct(fct, dataset, append_missing=True)
        print("%s Accuracy: %f" % (name, baseline_acc))
        print("%s Precision: %f" % (name, baseline_prec))


    outdir=os.path.join(args.output_dir, args.run_name, "eval")
    outfile = os.path.join(outdir, "results-%s.txt"%args.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outfile, 'w') as writer:
        writer.write("Model Accuracy: %f\n" % model_acc)
        writer.write("Model Precision: %f\n" % model_prec)
        for name in baseline_fcts.keys():
            writer.write("%s Accuracy: %f\n" % (name, baseline_acc))
            writer.write("%s Precision: %f\n" % (name, baseline_prec))
