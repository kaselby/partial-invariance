
import torch
import os
import tqdm
import numpy as np
import pandas as pd
import argparse
from transformers import BertTokenizer

from generators import CaptionGenerator, bert_tokenize_batch
from train_coco import evaluate, load_flickr_data, load_coco_data

use_cuda=torch.cuda.is_available()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

sizes=[5, 15, 30, 50]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default='final-runs/coco')
    parser.add_argument('--set_size', type=int, nargs=2, default=[10, 30])
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--steps', type=int, default=500
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--data_dir', type=str, default='data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_kwargs={
        'set_size':(3, 10),
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
    

    model_dir = os.path.join(args.basedir, args.run_name)
    runs = get_runs(model_dir)
    accs = torch.zeros(len(runs))
    for i, run_num in enumerate(runs):
        model_path = os.path.join(model_dir, run_num, 'model.pt')
        if not os.path.exists(model_path):
            break
        model = torch.load(model_path)
        accs[i] = evaluate(model, test_generator, args.steps, batch_size=args.batch_size, data_kwargs=data_kwargs)
    avg_acc = accs.mean()
    std = accs.std()
    
    with open(args.outfile, 'a') as writer:
        writer.write("%s: \tAvg:%f\tStdev:%f\tAccs:%s" % (args.run_name, avg_acc.item(), std.item(), str(accs.tolist())))

                        



