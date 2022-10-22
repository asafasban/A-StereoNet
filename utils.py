
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None and tag.find('weight') > 0:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

def plotData(data, labels, rows, columns, filename=None):
    r = rows
    c = columns
    if rows < 2:
        r += 1
    if columns < 2:
        c += 1

    fig, ax = plt.subplots(r, c)
    fig.set_size_inches(20, 20)
    for i in range(0, rows):
        for j in range(0, columns):
            value = data[rows * i + j]
            if not type(value).__module__ == np.__name__:
                value = value[0].permute(1, 2, 0).cpu().detach().numpy()
            else:
                value = value.squeeze()

            label = labels[rows * i + j]
            ax[i, j].imshow(value)
            ax[i, j].set_title(label)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()