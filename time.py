import timeit
import os
from utils import *
from generators import *
from models2 import *
import torch

#run_name="w_24-40_gmm"

n=16
N=300
sizes = torch.linspace(2.5,7,8).exp().round().int()

generator=GaussianGenerator(num_outputs=2)
#model = torch.load(os.path.join("runs", run_name, "model.pt"))
models = {
    'base': MultiSetTransformer(n, n*2, n*4, 1, equi=False, nn_attn=False).cuda(),
    'nn_attn': MultiSetTransformer(n, n*2, n*4, 1, equi=False, nn_attn=True).cuda(),
    'equi:': MultiSetTransformer(n, 16, 32, 1, equi=True, nn_attn=False).cuda(),
    'nn_attn-equi': MultiSetTransformer(n, 16, 32, 1, equi=True, nn_attn=True).cuda()
}
baseline = kl_knn

def test(generator, fct, bs=32, **kwargs):
    X = generator(bs, **kwargs)
    out = fct(*X)

results={}
for s in sizes:
    size = s.item()
    #print("Size: ", size)
    results[size]={}
    for model_name, model in models.items():
        t_model = timeit.timeit(lambda: test(generator, model, n=n, set_size=(size, int(size*1.5))), number=N)
        results[size][model_name] = t_model
    t_baseline = timeit.timeit(lambda: test(generator, baseline, n=n, set_size=(size, int(size*1.5))), number=N)
    results[size]['baseline'] = t_baseline

    print("%d:" % size)
    print("\tbaseline: %f" % results[size]['baseline'])
    for model_name, model in models.items():
        print("\t%s: %f" % (model_name, results[size][model_name]))

    
        