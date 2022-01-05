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
        return torch.stack([dataset[indices[i]] for i in range(N)], dim=0).unsqueeze(0)
    X = sample(dataset1, set_size)
    Y = sample(dataset2, set_size)
    out = model(X.to(device), Y.to(device))
    dist = -1 * F.logsigmoid(out)[0].item()
    return dist


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default='VLCS')
parser.add_argument('--output_dir', type=str, default='final-runs/DomainBed/')
parser.add_argument('--num_samples', type=int, default=500)
parser.add_argument('--num_sets', type=int, default=5)
parser.add_argument('--img_size', type=int, default=224)

args = parser.parse_args()

device = torch.device("cuda")

dataset_cls = vars(datasets)[args.dataset]
n_envs = len(dataset_cls.ENVIRONMENTS)
test_envs = list(range(n_envs))
dataset = dataset_cls(args.data_dir, test_envs, False, args.img_size)


model = torch.load(args.model_path)

table=[]
for i, source_name in enumerate(dataset_cls.ENVIRONMENTS):
    record = [source_name]
    print("Source:", source_name)
    for j, target_name in enumerate(dataset_cls.ENVIRONMENTS):
        dists = []
        print("Target:", target_name)
        for k in range(args.num_sets):
            dist_ijk = predict(model, dataset[i], dataset[j], args.num_samples, device)
            dists.append(dist_ijk)
            print("Distance:", str(dist_ijk))
        record.append(sum(dists)/len(dists))
    table.append(record)

results = tabulate.tabulate(table, headers=dataset_cls.ENVIRONMENTS, tablefmt='rst')
print(results)

output_dir = os.path.join(args.output_dir, args.dataset)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir,"results.txt"), 'w') as outfile:
    outfile.write(results)



