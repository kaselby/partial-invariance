import torch
import csv
import os 

import domainbed.datasets

def corr(l1, l2):
    return ( (l1-l1.mean(dim=1, keepdim=True)) * (l2-l2.mean(dim=1, keepdim=True)) ).mean(dim=1) / l1.std(unbiased=False, dim=1) / l2.std(unbiased=False, dim=1)

def load_csv(path):
    with open(path, 'r') as reader:
        csvreader = csv.reader(reader, delimiter=',')
        lines = [[float(x) for x in line] for line in csvreader]
    return torch.tensor(lines)

domainbed_dir = "/h/kaselby/DomainBed/final-results"
model_dir = "final-runs/DomainBed"
model_file="tg1_results.csv"

algorithm="ERM"
datasets=["VLCS", "OfficeHome"]
n_trials=3

for dataset in datasets:
    generalization_results = load_csv(os.path.join(domainbed_dir, "summary", "%s_%s_summary.csv" % (dataset, algorithm)))
    model_dists = load_csv(os.path.join(model_dir, dataset, model_file))
    corrs_by_env = corr(model_dists, generalization_results)
    corr_all = corr(model_dists.view(1, -1), generalization_results.view(1, -1))
    print("Dataset:", dataset)
    for i, env in enumerate(vars(domainbed.datasets)[dataset].ENVIRONMENTS):
        print("\tDomain: %s\tCorr:%f" % (env, corrs_by_env[i].item()))
    print("Overall Corr:", corr_all.item())

