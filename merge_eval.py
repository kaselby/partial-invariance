from argparse import ArgumentParser
import os
import torch
import glob

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

model_suffixes = ['csab','pine','naive','rn','cross-only','sum-merge']

parser = ArgumentParser()
parser.add_argument('run_name', type=str)
parser.add_argument('--basedir', type=str, default='final-runs')
parser.add_argument('--dataset', type=str, default='mnist')
args = parser.parse_args()

base_dir = os.path.join(args.basedir, args.dataset)

results={}
for model_name in model_suffixes:
    model_dir = os.path.join(base_dir, args.run_name + "_" + model_name)
    if os.path.exists(model_dir):
        runs = get_runs(model_dir)
        accs = []
        for run_num in runs:
            run_dir = os.path.join(model_dir, run_num, "logs.pt")
            if os.path.exists(run_dir):
                logs = torch.load(run_dir)
                accs.append(logs['test_acc'])
        if len(accs) > 0:
            results[model_name] = {'all_accs': accs, 'avg_acc': sum(accs)/len(accs)}

output_file = os.path.join(base_dir, args.run_name +"_results.txt")
with open(output_file, 'w') as outfile:
    for model_name, model_results in results.items():
        outfile.write(model_name + "\tAll Accs:" + str(model_results['all_accs']) + "\tAvg Acc:" + str(model_results['avg_acc'])+"\n")


    







