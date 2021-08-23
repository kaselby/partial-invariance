from models import *
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tabulate
from timeit import default_timer as timer

set_sizes = (
    (10,15),
    (30,45),
    (100,150),
    (300,450),
    (1000,1500),
    (3000,4500)
)
N = (2,3,5,10)
results = []
t1=timer()
for set_size in set_sizes:
  results_ss=[set_size,]
  for n in N:
    l1, l2 = evaluate(kl_knn, generate_multi_params(generate_gaussian_nd), kl_nd_gaussian, exact_loss=True, n=n, set_size=set_size)
    t2 = timer()
    print("N: %d, SS: %d, T: %f" % (n, set_size[0], t2-t1))
    t1 = t2
    results_ss.append((l1,l2))
  results.append(results_ss)


print(tabulate.tabulate(results, headers=N, tablefmt='rst'))
    