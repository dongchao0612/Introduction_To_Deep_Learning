import datetime

import numpy  as np
import matplotlib.pylab as plt
from mypackge.ch03 import sigmoid, softmax
from mypackge.ch04 import numerical_gradient, cross_entropy_error
from mypackge.mnist import load_mnist
import time

plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size)
        }

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)
        grand = {
            "W1": numerical_gradient(loss_w, self.params["W1"]),
            "b1": numerical_gradient(loss_w, self.params["b1"]),
            "W2": numerical_gradient(loss_w, self.params["W2"]),
            "b2": numerical_gradient(loss_w, self.params["b2"])
        }
        return grand


# 定义画图函数

def matplot_loss(train_loss_list, val_loss_list):
    plt.plot(train_loss_list, label="train_loss")
    plt.plot(val_loss_list, label="val_loss")
    plt.legend(loc="best")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("训练集和测试集loss对比图")
    plt.savefig("loss")
    plt.show()


def matplot_acc(train_acc_list, test_acc_list):
    plt.plot(train_acc_list, label="train_acc")
    plt.plot(test_acc_list, label="test_acc")
    plt.legend(loc="best")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.title("训练集和测试集acc对比图")
    plt.ylim(0, 1)
    plt.savefig("accuracy")
    plt.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)

    # 超参数
    iters_num = 500
    train_size = x_train.shape[0]
    batch_size = 5
    learn_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    net = TwoLayerNet(784, 5, 10)

    for i in range(iters_num):
        # if i % 100 == 0:
        print(f"第{i + 1}轮开始...")
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.numerical_gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            net.params[key] -= learn_rate * grad[key]

        # train_loss = net.loss(x_batch, t_batch)
        train_acc = net.accuracy(x_train, t_train)
        # test_loss=net.loss(x_test, t_test)
        test_acc = net.accuracy(x_test, t_test)

        # train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        # test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    # matplot_loss(train_loss_list, test_loss_list)
    matplot_acc(train_acc_list, test_acc_list)
