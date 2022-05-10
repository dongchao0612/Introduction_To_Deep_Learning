# coding: utf-8
from collections import OrderedDict
import matplotlib.pylab as plt
import numpy as np

from mypackge.ch03 import sigmoid, softmax
from mypackge.ch04 import cross_entropy_error, numerical_gradient
from mypackge.ch05 import Affine, Relu, SoftmaxWithLoss

from mypackge.mnist import load_mnist
class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size)
        }

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        #print("predict")
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        #print("\tloss")
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads
if __name__ == '__main__':

    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    iters_num = 200
    train_size = x_train.shape[0]  # 60000
    learn_rate = 0.1
    batch_size = 10

    train_loss_list = []
    train_acc_list=[]
    test_acc_list=[]

    iter_per_epoch = max(train_size / batch_size, 1)

    net = TwoLayerNet(784, 5, 10)

    for i in range(iters_num):

        print(f"第{i + 1}轮开始...")

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads=net.num(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            net.params[key] -= learn_rate * grads[key]

        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print((loss))

        # #if i % iter_per_epoch == 0:
        # train_acc = net.accuracy(x_train, t_train)
        # test_acc = net.accuracy(x_test, t_test)
        # train_acc_list.append(train_acc)
        # test_acc_list.append(test_acc)
        # print("loss, train acc, test acc | " + str(loss) + ", " + str(train_acc * 100) + ", " + str(test_acc * 100))

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list, label='loss')
    # plt.plot(x, train_acc_list, label='train acc')
    # plt.plot(x, test_acc_list, label='test acc')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    #plt.ylim(0, 1.0)
    #plt.legend(loc='lower right')
    plt.savefig("loss_mini.png")
    plt.show()
