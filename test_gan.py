import torch
import os
import tqdm
import numpy as np
import pandas as pd

from md_generator import MetaDatasetGenerator, Split


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
    n_datasets = len(episode.datasets)
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
    n_datasets = len(episode.datasets)
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
                



device=torch.device("cuda")

basedir="final-runs"
dataset="meta-dataset/discriminator"

run_name = "baseline_3_naive/1"
image_size=84
episode_classes=200
episode_datasets=11
steps=250
batch_size=16

data_kwargs={
    'set_size':(10,30),
    'p_aligned': 0.5,
    'p_dataset': 0.3,
    'p_same': 0.5
}

model = torch.load(os.path.join(basedir, dataset, run_name, "model.pt"))
test_generator = MetaDatasetGenerator(image_size=image_size, split=Split.TEST, device=device)

episode = test_generator.get_episode(episode_classes, episode_datasets)
y,yhat, (dl, sd) = eval_disc(model, episode, steps, batch_size, data_kwargs)


(acc, dl_acc, dl_pos_acc, dl_neg_acc, cl_acc, cl_pos_acc, cl_neg_acc, cl_neg_sd_acc, cl_neg_dd_acc), _ = summarize_eval(y, yhat, dl, sd, return_all=True)

lines=[]
lines.append("Overall Acc: "+ str(acc))
lines.append("Dataset-level Acc: " + str(dl_acc))
lines.append("\tPositive Acc: " + str(dl_pos_acc))
lines.append("\tNegative Acc: " + str(dl_neg_acc))
lines.append("Class-level Acc: " + str(cl_acc))
lines.append("\tPositive Acc: " + str(cl_pos_acc))
lines.append("\tNegative Acc: " + str(cl_neg_acc))
lines.append("\t\tSame-Dataset Acc: " + str(cl_neg_sd_acc))
lines.append("\t\tCross-Dataset Acc: " + str(cl_neg_dd_acc))
for line in lines:
    print(line)

data_kwargs={
    'set_size':(6,10)
}
steps=16

accs = eval_by_dataset(model, test_generator, steps, batch_size, (10,30))

dists, accs = eval_cross_dataset(model, episode, steps)

out_path = os.path.join(basedir, dataset, run_name)

def save_csv(tensor, path):
    df=pd.Dataframe(tensor.numpy())
    df.to_csv(path, index=False)

save_csv(dists, os.path.join(out_path, "dataset_dists.csv"))
save_csv(dists, os.path.join(out_path, "dataset_accs.csv"))
