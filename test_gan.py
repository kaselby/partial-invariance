import torch
import os

from md_generator import MetaDatasetGenerator, Split


def eval_disc(model, episode, steps, batch_size, data_kwargs):
    N = batch_size * steps
    with torch.no_grad():
        y,yhat,dl,sd=[],[],[],[]
        for i in range(steps):
            (X,Y), target, (dataset_level, same_dataset) = episode(batch_size, eval=True, **kwargs)
            out = model(X,Y).squeeze(-1)
            y.append(target)
            yhat.append(out>0)
            dl.append(dataset_level)
            sd.append(same_dataset)
            #n_correct += torch.eq((out > 0), target).sum().item()
        y=torch.cat(y, dim=0)
        yhat=torch.cat(y, dim=0)
        dl=torch.cat(dl, dim=0)
        sd=torch.cat(sd, dim=0)
    return y, yhat, (dl, sd)

device=torch.device("cuda")

basedir="final-runs"
dataset="meta-dataset/discriminator"

run_name = "test_gan_1"
image_size=84
episode_classes=200
episode_datasets=11

data_kwargs={
    'set_size':(6,10),
    'p_aligned': 0.5,
    'p_dataset': 0.3,
    'p_sameset': 0.5
}

model = torch.load(os.path.join(basedir, dataset, run_name, "model.pt"))
test_generator = MetaDatasetGenerator(image_size=image_size, split=Split.TEST, device=device)

episode = test_generator.get_episode(episode_classes, episode_datasets)
y,yhat = eval_disc(model, episode, 200, 16, **data_kwargs)

