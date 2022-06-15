import torch
import math
import os
from models2 import MultiSetTransformer, PINE

basenum=4962598
folder="final-runs/mi/mi_10_corr_updated_fixed"

for i in range(basenum, basenum+9):
    checkpoint1=torch.load("/checkpoint/kaselby/%d/checkpoint.pt"%i)
    model=checkpoint1['model']._modules['module']
    outfile = folder
    if type(model) is MultiSetTransformer:
        if model.equi:
            outfile = outfile + "_equi"
    else:
        outfile = outfile + "_pine"
    j = int(math.floor((i-basenum)/3))
    outfile = os.path.join(outfile, str(j))
    if not os.path.exists(outfile):
        os.makedirs(outfile)
    torch.save(model, os.path.join(outfile,"model.pt"))
    del checkpoint1, model
