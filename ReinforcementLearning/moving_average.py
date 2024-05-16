import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def moving_average(dataset, width):
    return [np.average(dataset[0:i]) for i in range(1,width)]+[np.average(dataset[i:i+width]) for i in range(len(dataset)-width+1)]

def plot_movingAverage(dataset, name, width, path_to_save):
    y = moving_average(dataset, width)
    x = np.arange(len(y))
    #plt.plot(x, dataset, 'k.-', label='Original data')
    plt.plot(x, moving_average(dataset, width), 'r.-', label='Running average')
    #plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.grid(linestyle=':')
    plt.legend()
    try:
        plt.savefig(path_to_save+name)
    except:
        os.makedirs(path_to_save)
        plt.savefig(path_to_save + name)
    plt.close()






