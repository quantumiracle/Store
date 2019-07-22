import gym
import numpy as np
import matplotlib.pyplot as plt


def plot_deform(min_idx, max_idx):

    x=np.array([5,7,9,11,13,15,
    4,6,8,10,12,14,16,
    3,5,7,9,11,13,15,17,
    2, 4,6,8,10,12,14,16,18,
    1,3,5,7,9,11,13,15,17,19,
    0,2,4,6,8,10,12,14,16,18,20,
    1,3,5,7,9,11,13,15,17,19,
    2, 4,6,8,10,12,14,16,18,
    3,5,7,9,11,13,15,17,
    4,6,8,10,12,14,16,
    5,7,9,11,13,15
    ])

    y=np.array([
        10,10,10,10,10,10,
        9,9,9,9,9,9,9,
        8,8,8,8,8,8,8,8,
        7,7,7,7,7,7,7,7,7,
        6,6,6,6,6,6,6,6,6,6,
        5,5,5,5,5,5,5,5,5,5,5,
        4,4,4,4,4,4,4,4,4,4,
        3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,
        1,1,1,1,1,1,1,
        0,0,0,0,0,0
    ])

    plt.figure(figsize=(5,4))
    plt.scatter(x,y, c='b')
    plt.scatter(x[min_idx], y[min_idx], c='g')
    plt.scatter(x[max_idx], y[max_idx], c='r')
    plt.savefig('./deform.png')
    plt.show()
    plt.pause(0.1)


if __name__ == '__main__':
    plot_deform(2,23)