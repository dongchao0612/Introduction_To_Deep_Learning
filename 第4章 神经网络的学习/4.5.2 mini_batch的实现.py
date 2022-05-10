import numpy  as np
import matplotlib.pylab as plt
from mypackge.ch03 import sigmoid, softmax
from mypackge.ch04 import numerical_gradient, cross_entropy_error
from mypackge.mnist import load_mnist

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

    def accurancy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / float(x.shape[0])

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

def matplot_loss(train_loss_list):
    plt.plot(train_loss_list, label="train_loss")
    plt.legend(loc="best")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("训练集loss图")
    plt.savefig("loss_mini")
    plt.show()


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 超参数
    iters_num = 50
    train_size = x_train.shape[0]  # 60000
    learn_rate = 0.1
    batch_size = 10

    train_loss_list = []
    test_loss_list = []


    net = TwoLayerNet(784, 5, 10)

    for i in range(iters_num):
        print(f"第{i + 1}轮开始...")
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grads = net.numerical_gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            net.params[key] -= learn_rate * grads[key]

        train_loss = net.loss(x_batch, t_batch)
        train_loss_list.append(train_loss)

        print(train_loss)


    matplot_loss(train_loss_list)

