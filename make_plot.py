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
    args = parser.parse_args()

    for run_name in args.run_names:
        filename = os.path.join("runs", run_name, "logs.pt")
        logs = torch.load(filename)
        losses = logs['losses']

        plt.figure(run_name)
        plt.plot(losses)
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("Mean Squared Error")
        plt.yscale("log")
    
    plt.show()
