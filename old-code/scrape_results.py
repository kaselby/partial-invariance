import torch
import os

CHECKPOINT_DIR="/checkpoint/kaselby"
outdir="final-runs/mi"
run_name="test_mi_sizes_ss"

#models=["csab","pine","naive","rn", "cross-only", "sum-merge"]
sizes=[20, 50, 100, 200, 500]
n_runs=3

checkpoint_num=5928106
for size in sizes:
    for i in range(n_runs):
    
    #for model in models:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, str(checkpoint_num), "checkpoint.pt")
        outdir_i = os.path.join(outdir, run_name+str(size), str(i))
        model_path = os.path.join(outdir_i, "model.pt")
        log_path = os.path.join(outdir_i, "logs.pt")
        if not os.path.exists(outdir_i):
            os.makedirs(outdir_i)
        if os.path.exists(checkpoint_path) and not (os.path.exists(model_path) and os.path.exists(log_path)):
            checkpoint = torch.load(checkpoint_path)
            torch.save(checkpoint, os.path.join(outdir_i, "checkpoint.pt"))
            torch.save(checkpoint['model']._modules['module'], model_path)
        checkpoint_num += 1

'''
for i in range(n_runs):
    for model in models:
        log_path = os.path.join(outdir, run_name+"_"+model, str(i), "logs.pt")
        if os.path.exists(log_path):
            logs = torch.load(log_path)
            if 'eval_acc' in logs and 'test_acc' not in logs:
                logs['test_acc'] = logs['eval_acc']
                del logs['eval_acc']
                torch.save(logs, log_path)
'''

