import argparse
import os
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import torch
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_names', type=str, nargs='+')
    parser.add_argument('--basedir', type=str, default="final-runs")
    args = parser.parse_args()

    for run_name in args.run_names:
        filename = os.path.join(args.basedir, run_name, "logs.pt")
        logs = torch.load(filename)
        losses = logs['losses']

        plt.figure(run_name)
        plt.plot(losses)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Mean Squared Error")
        plt.yscale("log")
    
    plt.show()


import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from train_coco import load_caption_data

train,test=load_caption_data("coco/images","coco/annotations")

def show(i):
    img,text=train[i]
    print(text)
    plt.imshow(  img.permute(1, 2, 0)  )
    plt.show()