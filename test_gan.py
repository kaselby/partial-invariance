import torch
import os

from md_generator import MetaDatasetGenerator, Split


def eval_disc(model, episode, steps, batch_size, data_kwargs):
    N = batch_size * steps
    with torch.no_grad():
        y,yhat,dl,sd=[],[],[],[]
        for i in range(steps):
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
    #prec = (y & yhat).sum().item() / yhat.sum().item()

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
    cl_neg_sd_acc, n_cl_neg_sd = get_acc(~dl & y & sd)
    cl_neg_dd_acc, n_cl_neg_dd = get_acc(~dl & ~y & ~sd)

    #dl_prec = (dl & y & yhat).sum().item() / (dl & yhat).sum().item()
    #cl_prec = (~dl & y & yhat).sum().item() / (~dl & yhat).sum().item()

    return (acc, dl_acc, dl_pos_acc, dl_neg_acc, cl_acc, cl_pos_acc, cl_neg_acc, cl_neg_sd_acc, cl_neg_dd_acc), (N, n_dl, n_dl_pos, n_dl_neg, n_cl, n_cl_pos, n_cl_neg, n_cl_neg_sd, n_cl_neg_dd)


device=torch.device("cuda")

basedir="final-runs"
dataset="meta-dataset/discriminator"

run_name = "test_gan_1"
image_size=84
episode_classes=200
episode_datasets=11
steps=1000

data_kwargs={
    'set_size':(6,10),
    'p_aligned': 0.5,
    'p_dataset': 0.3,
    'p_same': 0.5
}

model = torch.load(os.path.join(basedir, dataset, run_name, "model.pt"))
test_generator = MetaDatasetGenerator(image_size=image_size, split=Split.TEST, device=device)

episode = test_generator.get_episode(episode_classes, episode_datasets)
y,yhat, (dl, sd) = eval_disc(model, episode, steps, 16, data_kwargs)


(acc, dl_acc, dl_pos_acc, dl_neg_acc, cl_acc, cl_pos_acc, cl_neg_acc, cl_neg_sd_acc, cl_neg_dd_acc), _ = summarize_eval(y, yhat, dl, sd, return_all=True)
