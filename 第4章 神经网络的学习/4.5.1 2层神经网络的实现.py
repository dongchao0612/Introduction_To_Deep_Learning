import numpy as np
from pprint import pprint
from mypackge.ch03 import sigmoid, softmax
from mypackge.ch04 import cross_entropy_error, numerical_gradient, cross_entropy_error_mini_onehot
from mypackge.mnist import load_mnist


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


if __name__ == '__main__':
    net = TwoLayerNet(784, 100, 10)

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_size = x_train.shape[0]  # 60000
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    y = net.predict(x_batch)

    # print(t_batch.shape, y.shape)  # (10, 10) (10, 10)
    grads = net.numerical_gradient(x_batch, t_batch)
    # print(grads['W1'].shape)
    print(net.loss(x_batch, t_batch))
