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

    basedir=os.path.join(args.basedir, args.target)


    N=100
    set_size=(100,150)
    rho = torch.tensor([-0.99,-0.9,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,0.9,0.99])
    if use_cuda:
        rho = rho.cuda()
    mi_true = mi_corr_gaussian(rho, d=args.n)
    results={}
    with torch.no_grad():
        mi_kraskov = torch.zeros_like(rho)
        for i in range(rho.size(0)):
            X, T = generator(N, n=args.n, corr=rho[i])
            mi_kraskov[i] = kraskov_mi1(*X).mean()
        results["kraskov"] = {"mean":(mi_kraskov - mi_true).abs().mean().item(),"std":0}

        run_paths = glob.glob(os.path.join(args.basedir, "mi", args.run_name+"*"))
        for run_path in run_paths:
            run_name = run_path.split("/")[-1]
            modeldir = os.path.join(args.basedir, "mi", run_name)
            all_runs = get_runs(modeldir)
            n_runs = 0
            mi_model = []
            for run_num in all_runs:
                model_path = os.path.join(modeldir, run_num, "model.pt")
                if os.path.exists(model_path):
                    model = torch.load(os.path.join(modeldir, run_num, "model.pt"))
                    model_args = torch.load(os.path.join(modeldir, run_num, "logs.pt"))['args']
                    mi_model_i = torch.zeros_like(rho)
                    for i in range(rho.size(0)):
                        X, T = generator(N, n=args.n, corr=rho[i])
                        out = model(*X).squeeze(-1)
                        if getattr(model_args, "scale_out", None) is not None:
                            if model_args.scale_out == "sq":
                                out = torch.pow(out, 2)
                            elif model_args.scale_out == "exp":
                                out = torch.exp(out)
                        mi_model_i[i] = out.mean()
                    mi_model.append(mi_model_i)
                    n_runs += 1
            mi_model = torch.stack(mi_model, dim=0)
            abs_error = (mi_model - mi_true).abs().mean(dim=1)
            results[run_name] = {"mean":abs_error.mean(dim=0).item(), "std":abs_error.std(dim=0).item()}
    
    print(results)


    

