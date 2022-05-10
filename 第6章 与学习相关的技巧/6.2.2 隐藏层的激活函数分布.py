import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

if __name__ == '__main__':
    x = np.random.randn(1000, 100)  # 1000个数据
    node_num = 100  # 节点个数
    hidden_layer_size = 5  # 层数

    activations = {}  # 激活值

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]
        #w = np.random.rand(node_num, node_num) *1
        #w = np.random.rand(node_num, node_num) *0.01
        w = np.random.rand(node_num, node_num) / np.sqrt(node_num)

        z = np.dot(x, w)

        a = sigmoid(z)
        activations[i] = a

        # print(a.shape)
        # break
    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(f"{i + 1}-layers")

        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()
