from utils import *
import os
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--basedir', type=str, default="final-runs")
    parser.add_argument('--n', type=int, default=2)

    return parser.parse_args()

def get_runs(run_name):
    subfolders = [f.name for f in os.scandir(run_name) if f.is_dir()]
    return subfolders

def get_mi(model, rho, set_size=(100,150), N=100, d=2):
    generator = CorrelatedGaussianGenerator(return_params=True)
    with torch.no_grad():
        mi = torch.zeros_like(rho)
        for i in range(rho.size(0)):
            X, T = generator(N, n=d, corr=rho[i], set_size=set_size)
            mi[i] = model(*X).mean()
    return mi

if __name__ == '__main__':
    args = parse_args()
    
    generator = CorrelatedGaussianGenerator(return_params=True)

    N=100
    set_size=(100,150)
    rho = torch.tensor([-0.99,-0.9,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,0.9,0.99])
    if use_cuda:
        rho = rho.cuda()
    mi_true = mi_corr_gaussian(rho, d=args.n)
    mi_model = torch.zeros_like(rho)
    mi_kraskov = torch.zeros_like(rho)
    with torch.no_grad():
        for i in range(rho.size(0)):
            X, T = generator(N, n=args.n, corr=rho[i])
            mi_kraskov[i] = kraskov_mi1(*X).mean()
        modeldir = os.path.join(args.basedir, "mi", args.run_name)
        all_runs = get_runs(modeldir)
        for run_num in all_runs:
            model = torch.load(os.path.join(modeldir, run_num, "model.pt"))
            model_args = torch.load(os.path.join(modeldir, run_num, "logs.pt"))['args']
            for i in range(rho.size(0)):
                X, T = generator(N, n=args.n, corr=rho[i])
                out = model(*X).squeeze(-1)
                if model_args.scale_out == "sq":
                    out = torch.pow(out, 2)
                elif model_args.scale_out == "exp":
                    out = torch.exp(out)
                mi_model[i] += out.mean()
            mi_model /= len(all_runs)


    torch.save({'rho':rho.cpu(), 'true':mi_true.cpu(), 'model':mi_model.cpu(), 'kraskov':mi_kraskov.cpu()}, 
        os.path.join(args.basedir, "mi", args.run_name, "rho.pt"))

