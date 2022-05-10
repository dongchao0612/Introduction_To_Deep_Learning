import numpy as np

from mypackge.ch06 import SGD, TwoLayerNet
from mypackge.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    train_size = x_train.shape[0]  # 60000
    iters_num = 10000
    batch_size = 100
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)  # 600

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    optimizer = SGD()

    for i in range(iters_num):

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grads = network.gradient(x_batch, t_batch)
        # print(grads)
        params = network.params
        optimizer.update(params, grads)
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            print("i = ", i)
            # print("**************************")
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train_acc", train_acc, test_acc)
