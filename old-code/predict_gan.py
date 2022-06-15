import argparse
import os
import torch
import torch.nn.functional as F
import math
import tabulate
import csv
import glob
import tqdm
import pandas as pd

import domainbed.datasets as datasets

def predict(model, dataset1, dataset2, set_size, device):
    def sample(dataset, N):
        N = min(N, len(dataset))
        indices = torch.randperm(len(dataset))
        return torch.stack([dataset[indices[i]][0] for i in range(N)], dim=0).unsqueeze(0)
    with torch.no_grad():
        X = sample(dataset1, set_size)
        Y = sample(dataset2, set_size)
        out = model(X.to(device), Y.to(device))
        dist = -1 * F.logsigmoid(out)[0].item()
    return dist

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

def save_csv(tensor, path):
    df=pd.Dataframe(tensor.numpy())
    df.to_csv(path, index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, default='final-runs/meta-dataset/discriminator')
parser.add_argument('--data_dir', type=str, default='/h/kaselby/DomainBed/domainbed/data')
parser.add_argument('--dataset', type=str, default='VLCS')
parser.add_argument('run_name', type=str)
parser.add_argument('--output_dir', type=str, default='final-runs/DomainBed/')
parser.add_argument('--num_samples', type=int, default=500)
parser.add_argument('--num_sets', type=int, default=100)
parser.add_argument('--img_size', type=int, default=224)

args = parser.parse_args()

device = torch.device("cuda")

dataset_cls = vars(datasets)[args.dataset]
n_envs = len(dataset_cls.ENVIRONMENTS)
test_envs = list(range(n_envs))
dataset = dataset_cls(args.data_dir, test_envs, False, args.img_size)

model_path = os.path.join(args.basedir, args.run_name)
all_runs = get_runs(model_path)
dists = torch.zeros(n_envs, n_envs)
for run_num in all_runs:
    run_path = os.path.join(model_path, run_num, 'model.pt')
    model = torch.load(run_path)
    for i, source_name in enumerate(dataset_cls.ENVIRONMENTS):
        for j, target_name in enumerate(dataset_cls.ENVIRONMENTS):
            dists_ij = []
            for k in tqdm.tqdm(range(args.num_sets)):
                dist_ijk = predict(model, dataset[i], dataset[j], args.num_samples, device)
                dists_ij.append(dist_ijk)
            dists[i, j] += sum(dists_ij)/len(dists_ij)
dists /= len(all_runs)


output_dir = os.path.join(args.output_dir, args.dataset)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = args.run_name + "_results.csv"
output_path = os.path.join(output_dir, output_file) 

save_csv(dists, output_path)


'''
table=[]
for i, source_name in enumerate(dataset_cls.ENVIRONMENTS):
    record = []#[source_name]
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

tablestr = tabulate.tabulate(table, headers=dataset_cls.ENVIRONMENTS, tablefmt='rst')
print(tablestr)

output_dir = os.path.join(args.output_dir, args.dataset)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = args.model_prefix + "_results.csv" if args.model_prefix is not None else "results.csv"
output_path = os.path.join(output_dir, output_file) 

with open(output_path, 'w') as writer:
    csvwriter = csv.writer(writer, delimiter=',')
    #csvwriter.writerow([""] + dataset_cls.ENVIRONMENTS)
    for line in table:
        csvwriter.writerow(line)
'''

'''
import os
import csv
import torch

basedir="final-runs/DomainBed/"

with open(os.path.join(basedir, "VLCS", "tg1_results.csv"), 'r') as reader:
    csvreader=csv.reader(reader,delimiter=',')
    tg1_vlcs_results = [line for line in csvreader]
    

with open('/h/kaselby/DomainBed/results/summary/VLCS_ERM_summary.csv', 'r') as reader:
    csvreader=csv.reader(reader,delimiter=',')
    vlcs_gen_results = [line for line in csvreader]


dists=[]
for line in tg1_vlcs_results:
    dists.append([float(x) for x in line[1:]])

dists = torch.tensor(dists)
'''