from collections import OrderedDict
import numpy as np
from mypackge.ch04 import numerical_gradient
from mypackge.ch05 import Affine, Relu, SoftmaxWithLoss


class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            # "W1": np.array([[-4.93191307e-06, -6.53995473e-03], [-3.05748125e-03, 2.38054047e-03],
            #                 [4.15106978e-03, 3.46450797e-03]]),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            # "W2": np.array([[-0.00183145, -0.00165298, -0.00897353], [0.00468271, -0.01084557, -0.00396773]]),
            "b2": np.zeros(output_size)
        }
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        # print("predict")
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

        # x:输入数据, t:监督数据

    def loss(self, x, t):
        # print("\tloss")
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        # print(layers)

        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads


if __name__ == "__main__":  # 测试
    net = TwoLayerNet(input_size=3, hidden_size=2, output_size=3)
    # print(net)
    # print("net.params**************")
    # print(net.params['W1'])
    # print(net.params['b1'])
    # print(net.params['W2'])
    # print(net.params['b2'])
    # print("net.layers**************")
    # print(net.layers["Affine1"].W,net.layers["Affine1"].b)
    # print(net.layers["Relu1"])
    # print(net.layers["Affine2"].W,net.layers["Affine2"].b)
    # print(net.lastLayer)
    #
    #
    x = np.array([[8, 1, 3],
                  [2, 0, 4]])
    t = np.array([[0, 1, 0],
                  [1, 0, 0]])  # one-hot
    # y = net.predict(x)
    # loss = net.loss(x, t)
    # print(loss)
    # grads = net.numerical_gradient(x, t)
    # print(grads)

    grads = net.gradient(x, t)  # 计算梯度
    # print(grads['W1'].shape)
    # print(grads['b1'].shape)
    # print(grads['W2'].shape)
    # print(grads['b2'].shape)
    print(grads)
