import numpy as np

from mypackge.ch03 import softmax
from mypackge.ch04 import cross_entropy_error_mini_onehot, numerical_gradient,cross_entropy_error_mini_labels


class simpleNet():
    def __init__(self):
        self.W = np.array([[0.47, 0.99, 0.84], [0.85, 0.03, 0.69]])

    def pridict(self, x):
        z = np.dot(x, self.W)
        return softmax(z)

    def loss(self, x, t):
        y = self.pridict(x)

        #loss = cross_entropy_error_mini_onehot(y, t)
        loss=cross_entropy_error_mini_labels(y, t)
        return loss


def f(w):
    return net.loss(x, t)


if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.pridict(x)

    t = np.array([2])
    print("x = ", x, "p = ", p, "t = ", t)
    los = net.loss(x, t)
    print(los)
    dw = numerical_gradient(f, net.W)
    print(dw)
