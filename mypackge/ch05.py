import numpy as np

from mypackge.ch03 import softmax
from mypackge.ch04 import cross_entropy_error_mini_onehot, cross_entropy_error


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        #print("\tRelu forward")
        self.mask = (x <= 0)
        # print("self.mask",self.mask)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        #print("\tRelu backward")
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backword(self, dout):
        dx=dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        # print(W.shape,b.shape)

    def forward(self, x):
        #print("\tAffine forward")
        self.x = x
        # print(x.shape)
        out = np.dot(x, self.W) + self.b
        #print(np.dot(x, self.W).shape)

        return out

    def backward(self, dout):
        #print("\tAffine backward")
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        #print("\tSoftmaxWithLoss forward")
        self.t = t
        self.y = softmax(x)
        #print("\t\tself.t:",self.y)
        self.loss = cross_entropy_error_mini_onehot(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        #print("\tSoftmaxWithLoss backward")
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
