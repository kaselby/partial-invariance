import timeit
import os
from utils import *
from generators import *
from models2 import *
import torch

#run_name="w_24-40_gmm"

n=16
sizes = torch.linspace(2.5,8,15).exp().round().int()

generator=GaussianGenerator(num_outputs=2)
#model = torch.load(os.path.join("runs", run_name, "model.pt"))
model1 = MultiSetTransformer(n, n*2, n*4, 1, equi=False, nn_attn=False).cuda()
model2 = MultiSetTransformer(n, n*2, n*4, 1, equi=False, nn_attn=True).cuda()
model3 = MultiSetTransformer(n, 16, 32, 1, equi=True, nn_attn=False).cuda()
model4 = MultiSetTransformer(n, 16, 32, 1, equi=True, nn_attn=True).cuda()
models = [model1, model2, model3, model4]
baseline = kl_knn

def test(generator, fct, bs=32, **kwargs):
    X = generator(bs, **kwargs)
    out = fct(*X)

results={}
for s in sizes:
    size = s.item()
    print("Size: ", size)
    results[size]={'baseline':0, 'models':[]}
    for model in models:
        t_model = timeit.timeit(lambda: test(generator, model, n=n, set_size=(size, int(size*1.5))), number=500)
        results[size]['models'].append(t_model)
    t_baseline = timeit.timeit(lambda: test(generator, baseline, n=n, set_size=(size, int(size*1.5))), number=500)
    results[size]['baseline'] = t_baseline


for s in sizes:
    size = s.item()
    print("%d:\n" % size)
    print("\tbaseline: %f\n" % results[size]['baseline'])
    for i,model in enumerate(models):
        print("\model %d: %f\n" % (i, results[size]['models'][i]))

        