import argparse
import os
import torch
import torch.nn.functional as F
import math
import tabulate

import domainbed.datasets as datasets

def predict(model, dataset1, dataset2, num_samples, device):
    def sample(dataset, N):
        N = min(N, len(dataset))
        indices = torch.randperm(len(dataset))
        return torch.stack([dataset[indices[i] for i in range(N)], dim=0).unsqueeze(0)
    X = sample(dataset1, set_size)
    Y = sample(dataset2, set_size)
    out = model(X.to(device), Y.to(device))
    dist = -1 * F.logsigmoid(out)[0].item()
    return dist


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--num_samples', type=int)
parser.add_argument('--num_sets', type=int)

args = parser.parse_args()

device = torch.device("cuda")

dataset_cls = vars(datasets)[args.dataset]
n_envs = len(dataset_cls.ENVIRONMENTS)
test_envs = list(range(n_envs))
dataset = dataset_cls(args.data_dir, test_envs, None)


model = torch.load(args.model_path)

table=[]
for i, source_name in dataset_cls.ENVIRONMENTS:
    record = [source_name]
    for j, target_name in dataset_cls.ENVIRONMENTS:
        dists = []
        for _ in range(args.num_sets):
            dists.append(predict(model, dataset[i], dataset[j], args.num_samples, device))
        record.append(sum(dists)/len(dists))
    table.append(record)

results = tabulate.tabulate(table, headers=dataset_cls.ENVIRONMENTS, tablefmt='rst')
print(results)

with open(args.output_path, 'w') as outfile:
    outfile.write(results)



