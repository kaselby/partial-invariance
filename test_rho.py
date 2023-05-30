
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.distributions import CorrelatedGaussianGenerator, CorrelatedGaussianGenerator2
from utils import mi_corr_gaussian, kl_mc

def load_run(run_path):
    args = torch.load(os.path.join(run_path, "args.pt"))['args']
    model = torch.load(os.path.join(run_path, "model.pt"))
    task = TASKS[args.task](args)
    train_dataset, val_dataset, test_dataset = task.build_dataset()
    device=torch.device("cuda")
    trainer = task.build_trainer(model, None, None, train_dataset, val_dataset, test_dataset, device, None, checkpoint_dir=None)
    
    return args, model, task, (train_dataset, val_dataset, test_dataset), trainer


def KL_estimate(X, Y):
    return X.sum(dim=1)/X.size(1) - Y.logsumexp(dim=1) + math.log(Y.size(1))

def forward_KL(model, X, Y):
    X0,X1 = X.chunk(2, dim=1)
    Y0,Y1 = Y.chunk(2, dim=1)
    X_out, Y_out = model(X0, Y0, X1, Y1)

    return _KL_estimate(X_out, Y_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--set_size', type=int, default=500)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--basedir', type=str, default='oct-runs')
    args = parser.parse_args()

    model_args, model, task, (train_dataset, val_dataset, test_dataset), trainer = load_run(os.path.join(basedir, run_name))

    gen2=CorrelatedGaussianGenerator2(return_params=True)

    rhos = torch.tensor([-0.99,-0.9,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,0.9,0.99]).cuda()
    mi_true = mi_corr_gaussian(rhos, d=model_args.n)
    mi_model = torch.zeros(rhos.size(0))
    for i, rho in enumerate(rhos):
        n_runs = args.n // bs
        outputs = torch.zeros(args.n)
        with torch.no_grad():
            for i in tqdm.tqdm(range(n_runs)):
                (X,Y), theta = self.train_dataset(args.bs, set_size=(set_size, set_size+1), **model_args['sample_kwargs'])
                model_out = trainer._forward(X,Y)
                outputs[i*bs:(i+1)*bs] = model_out.cpu()
        mi_model[i] = outputs.mean()

    torch.save({'rho':rho.cpu(), 'true':mi_true.cpu(), 'model':mi_model.cpu()}, 
        os.path.join(args.basedir, args.run_name, "rho.pt"))

