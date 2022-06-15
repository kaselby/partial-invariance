import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import torch

dict1 = torch.load("partial-invariance/final-runs2/mi/baseline_feb10_equi/rho.pt")
dict2 = torch.load("mine-pytorch/mine_rho_2.pt")

plt.figure()

plt.plot(dict1['rho'], (dict1['model'] - dict1['true']).abs(), label='ours')
plt.plot(dict1['rho'], (dict1['kraskov'] - dict1['true']).abs(), label='kraskov')
plt.plot(dict1['rho'], (torch.tensor([x.item() for x in dict2['mine']]) - dict1['true']).abs(), label='mine')
plt.xlabel('correlation')
plt.ylabel('absolute error')

plt.legend()
plt.show()

'''
plt.plot(dict1['rho'], dict1['true'], label='true')
plt.plot(dict1['rho'], dict1['model'], label='ours')
plt.plot(dict1['rho'], dict1['kraskov'], label='kraskov')
plt.plot(dict1['rho'], dict2['mine'], label='mine')
plt.xlabel('correlation')
plt.ylabel('mi')
'''


'''
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import torch
sizes = torch.linspace(2.5,8,15).exp().round().int()
dict = torch.load("final-runs/kl/kl_2_baseline/ss_losses.pt")

plt.figure()
for k,v in dict1.items():
    plt.plot(sizes, v, label=k)
plt.xlabel('set size')
plt.ylabel('mean absolute error')
plt.legend()
plt.show()

plt.plot(sizes, dict1['true'], label='true')
plt.plot(dict1['rho'], dict1['model'], label='ours')
plt.plot(dict1['rho'], dict1['kraskov'], label='kraskov')
plt.plot(dict1['rho'], dict2['mine'], label='mine')
plt.xlabel('correlation')
plt.ylabel('mi')
plt.legend()
plt.show()
'''