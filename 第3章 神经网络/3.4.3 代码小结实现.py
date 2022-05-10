import numpy as np

from mypackge.ch03 import sigmoid, identity_func


def init_network():
    network = {
        "W1": np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]),
        "B1": np.array([0.1, 0.2, 0.3]),

        "W2": np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]),
        "B2": np.array([0.1, 0.2]),

        "W3": np.array([[0.1, 0.3], [0.2, 0.4]]),
        "B3": np.array([0.1, 0.2]),
    }
    return network


def forward(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["B1"], network["B2"], network["B3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    z3 = identity_func(a3)

    return z3


if __name__ == '__main__':
    network = init_network()
    # print(network["B1"].shape)#(3,)
    x = np.array([[1.0, 0.5]])
    y = forward(network, x)
    print("y = ", y)
