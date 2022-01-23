
import torch
import os
import tqdm
import numpy as np
import pandas as pd
import argparse

from md_generator import MetaDatasetGenerator, Split



use_cuda=torch.cuda.is_available()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

def save_csv(tensor, path):
    df=pd.DataFrame(tensor.numpy())
    df.to_csv(path, index=False)

def print_accs(accs):
    lines = []
    lines.append("Overall Acc: "+ str(accs[0].item()))
    lines.append("Dataset-level Acc: " + str(accs[1].item()))
    lines.append("\tPositive Acc: " + str(accs[2].item()))
    lines.append("\tNegative Acc: " + str(accs[3].item()))
    lines.append("Class-level Acc: " + str(accs[4].item()))
    lines.append("\tPositive Acc: " + str(accs[5].item()))
    lines.append("\tNegative Acc: " + str(accs[6].item()))
    lines.append("\t\tSame-Dataset Acc: " + str(accs[7].item()))
    lines.append("\t\tCross-Dataset Acc: " + str(accs[8].item()))
    return "\n".join(lines)

def eval_disc(model, episode, steps, batch_size, data_kwargs):
    N = batch_size * steps
    with torch.no_grad():
        y,yhat,dl,sd=[],[],[],[]
        for i in tqdm.tqdm(range(steps)):
            (X,Y), target, (dataset_level, same_dataset) = episode(batch_size, eval=True, **data_kwargs)
            out = model(X,Y).squeeze(-1)
            y.append(target)
            yhat.append(out>0)
            dl.append(dataset_level)
            sd.append(same_dataset)
            #n_correct += torch.eq((out > 0), target).sum().item()
        y=torch.cat(y, dim=0)
        yhat=torch.cat(yhat, dim=0)
        dl=torch.cat(dl, dim=0)
        sd=torch.cat(sd, dim=0)
    return y, yhat, (dl, sd)

def summarize_eval(y, yhat, dl, sd, return_all=False):
    N = y.size(0)
    correct = y==yhat
    acc = (y==yhat).sum().item() / N
    if not return_all:
        return acc
    def get_acc(labels):
        n = labels.sum().item()
        return (labels & correct).sum().item() / n, n
    dl_acc, n_dl = get_acc(dl)
    dl_pos_acc, n_dl_pos = get_acc(dl & y)
    dl_neg_acc, n_dl_neg = get_acc(dl & ~y)
    cl_acc, n_cl = get_acc(~dl)
    cl_pos_acc, n_cl_pos = get_acc(~dl & y)
    cl_neg_acc, n_cl_neg = get_acc(~dl & ~y)
    cl_neg_sd_acc, n_cl_neg_sd = get_acc(~dl & ~y & sd)
    cl_neg_dd_acc, n_cl_neg_dd = get_acc(~dl & ~y & ~sd)
    #dl_prec = (dl & y & yhat).sum().item() / (dl & yhat).sum().item()
    #cl_prec = (~dl & y & yhat).sum().item() / (~dl & yhat).sum().item()
    return (acc, dl_acc, dl_pos_acc, dl_neg_acc, cl_acc, cl_pos_acc, cl_neg_acc, cl_neg_sd_acc, cl_neg_dd_acc), (N, n_dl, n_dl_pos, n_dl_neg, n_cl, n_cl_pos, n_cl_neg, n_cl_neg_sd, n_cl_neg_dd)


def eval_by_dataset(model, dataset, steps, batch_size, set_size):
    n_datasets = dataset.N
    with torch.no_grad():
        accs = torch.zeros(n_datasets)
        for i in range(n_datasets):
            episode = dataset.get_dataset(i)
            for _ in range(steps):
                (X, Y), target = episode(batch_size, dataset_id=0, set_size=set_size)
                out = model(X,Y).squeeze(-1)
                accs[i] += ((out > 0) == target).float().sum().cpu()
            accs[i] /= (steps * batch_size)
    return accs
                 


def eval_cross_dataset(model, dataset, steps, batch_size, set_size, classes_per_dataset=100):
    n_datasets = dataset.N
    with torch.no_grad():
        dists = torch.zeros(n_datasets, n_datasets)
        accs = torch.zeros(n_datasets, n_datasets)
        for i in range(n_datasets):
            for j in range(n_datasets):
                episode = dataset.get_episode_from_datasets((i,j), classes_per_dataset)
                for _ in range(steps):
                    X, Y = episode.compare_datasets(0, 1, batch_size=batch_size, set_size=set_size)
                    out = model(X,Y).squeeze(-1).cpu()
                    dist = -1 * F.logsigmoid(out)[0].sum()
                    acc = ((out > 0) == (i==j)).float().sum()
                    dists[i][j] += dist
                    accs[i][j] += acc
                dists[i][j] /= (steps * batch_size)
                accs[i][j] /= (steps * batch_size)
    return dists, accs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default='final-runs/meta-dataset/discriminator')
    parser.add_argument('--set_size', type=int, nargs=2, default=[10, 30])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_eval_steps', type=int, default=500)
    parser.add_argument('--dataset_eval_steps', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=84)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_kwargs={
        'set_size':(10,30),
        'p_aligned': 0.5,
        'p_dataset': 0.3,
        'p_same': 0.5
    }

    device=torch.device("cuda")

    test_generator = MetaDatasetGenerator(image_size=args.image_size, split=Split.TEST, device=device)

    data_kwargs={
        'set_size':args.set_size,
        'p_aligned': 0.5,
        'p_dataset': 0.3,
        'p_same': 0.5
    }
    episode_classes = 200
    episode_datasets=11
    

    model_dir = os.path.join(args.basedir, args.run_name)
    runs = get_runs(model_dir)
    accs = torch.zeros(len(runs), 9)
    dataset_cl_accs = torch.zeros(len(runs), test_generator.N)
    #dataset_cross_accs = torch.zeros(len(runs), test_generator.N, test_generator.N)
    for i, run_num in enumerate(runs):
        model_path = os.path.join(model_dir, run_num, 'model.pt')
        if not os.path.exists(model_path):
            break
        model = torch.load(model_path)
        episode = test_generator.get_episode(episode_classes, episode_datasets)
        y,yhat, (dl, sd) = eval_disc(model, episode, args.base_eval_steps, args.batch_size, data_kwargs)
        accs[i,:] = torch.tensor(summarize_eval(y, yhat, dl, sd, return_all=True)[0])
        del episode

        dataset_cl_accs[i,:] = eval_by_dataset(model, test_generator, args.dataset_eval_steps, args.batch_size, args.set_size)
        #dataset_cross_accs[i,:,:] = eval_cross_dataset(model, test_generator, args.dataset_eval_steps, args.batch_size, args.set_size)
    accs = accs.mean(dim=0)
    dataset_cl_accs = dataset_cl_accs.mean(dim=0)
    #dataset_cross_accs = dataset_cross_accs.mean(dim=0)

    outdir = os.path.join(args.basedir, args.run_name)
    with open(os.path.join(outdir, "analysis.txt"), 'w') as writer:
        writer.write(print_accs(accs))
    
    #save_csv(dataset_cross_accs, os.path.join(outdir, "dataset_dists.csv"))
    save_csv(dataset_cl_accs, os.path.join(outdir, "dataset_accs.csv"))

