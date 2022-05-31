
import torch
import os
import tqdm
import numpy as np
import pandas as pd
import argparse
import glob
from transformers import BertTokenizer

from generators import CaptionGenerator, bert_tokenize_batch
from train_coco import evaluate, load_flickr_data, load_coco_data

use_cuda=torch.cuda.is_available()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--eval_all', action='store_true')
    parser.add_argument('--basedir', type=str, default='final-runs2')
    parser.add_argument('--set_size', type=int, nargs=2, default=[10, 30])
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--outdir', type=str, default='evals')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--data_dir', type=str, default='data')
    return parser.parse_args()


def eval_model(model_dir, dataset, steps, bs, data_kwargs):
    runs = get_runs(model_dir)
    accs = torch.zeros(len(runs))
    for i, run_num in enumerate(runs):
        model_path = os.path.join(model_dir, run_num, 'model.pt')
        if not os.path.exists(model_path):
            break
        model = torch.load(model_path)
        accs[i] = evaluate(model, dataset, steps, batch_size=bs, data_kwargs=data_kwargs)
    avg_acc = accs.mean()
    std = accs.std()
    return accs, avg_acc, std


if __name__ == '__main__':
    args = parse_args()

    data_kwargs={
        'set_size': args.set_size,
    }

    device=torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenize_fct = bert_tokenize_batch
    tokenize_args = (tokenizer,)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset == "coco":
        train_dataset, val_dataset, test_dataset = load_coco_data(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "annotations"))
    else:
        train_dataset, val_dataset, test_dataset = load_flickr_data(os.path.join(dataset_dir, "images"), os.path.join(dataset_dir, "annotations.token"), os.path.join(dataset_dir, "splits.json"))
    test_generator = CaptionGenerator(test_dataset, tokenize_fct, tokenize_args, device=device)
    

    if args.eval_all:
        run_paths = glob.glob(os.path.join(args.basedir, args.dataset, args.run_name+"*"))
        run_names = [run_path.split("/")[-1] for run_path in run_paths]

        results = {}
        for run_name in run_names:
            model_dir = os.path.join(args.basedir, args.dataset, run_name)
            accs, avg_acc, stdev = eval_model(model_dir, test_generator, args.steps, args.batch_size, data_kwargs)
            results[run_name] = {'accs': accs.tolist(), 'avg_acc': avg_acc.item(), 'stdev': stdev.item()}
        
        outfile = os.path.join(args.outdir, "%s_%d-%d_results.txt" % (args.run_name, *args.set_size)) 
        with open(outfile, 'a') as writer:
            for run_name, run_results in results.items():
                writer.write("%s: \tAvg:%f\tStdev:%f\tAccs:%s\n" % (run_name, run_results['avg_acc'], run_results['stdev'], str(run_results['accs'])))
    else:
        model_dir = os.path.join(args.basedir, args.dataset, args.run_name)
        accs, avg_acc, stdev = eval_model(model_dir, test_generator, args.steps, args.batch_size, data_kwargs)
        outfile = os.path.join(args.outdir, "%s_%d-%d_results.txt" % (args.run_name, *args.set_size))
        with open(outfile, 'a') as writer:
            writer.write("%s: \tAvg:%f\tStdev:%f\tAccs:%s" % (args.run_name, avg_acc.item(), std.item(), str(accs.tolist())))

