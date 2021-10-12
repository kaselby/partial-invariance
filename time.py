import timeit
import os
from utils import *
import torch

run_name="w_24-40_gmm"

generator=GaussianGenerator(num_outputs=2)
model = torch.load(os.path.join("runs", run_name, "model.pt"))
baseline = wasserstein

def test(generator, fct, bs=64, **kwargs):
    X = generator(bs, **kwargs)
    out = fct(*X)

t_model = timeit.timeit(lambda: test(generator, model, n=32), number=1000)
t_baseline = timeit.timeit(lambda: test(generator, baseline, n=32), number=1000)

print("Model:", t_model)
print("Baseline:", t_baseline)
        