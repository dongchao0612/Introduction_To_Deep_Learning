import pickle
import matplotlib.pylab as plt
import numpy as np
from PIL import Image

from mypackge.ch03 import sigmoid, softmax, relu
from mypackge.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    return x_train, t_train, x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb")as fp:
        network = pickle.load(fp)
    return network


def predict(network, x):
    w1, w2, w3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    # print("Z1.shape = ", z1.shape)#Z1.shape =  (50,)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    # print("Z2.shape = ", z2.shape)#Z2.shape =  (100,)

    a3 = np.dot(z2, w3) + b3
    z3 = softmax(a3)
    # print("Z3.shape = ", z3.shape)#Z3.shape =  (10,)

    return z3


if __name__ == '__main__':
    x_train, t_train, x_test, t_test = get_data()
    print(x_train[0].shape)  # (784,)
    print(t_train[0])  # 5
    exit(0)
    # print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)  # (60000, 784) (60000,) (10000, 784) (10000,)
    network = init_network()
    # print(network)
    # print(network['W1'].shape)  # (784, 50)
    # print(network['b1'].shape)  # (50,)
    # print(network['W2'].shape)  # (50, 100)
    # print(network['b2'].shape)  # (100,)
    # print(network['W3'].shape)  # (100, 10)
    # print(network['b3'].shape)  # (10,)

    '''
    (1,784)*(784, 50)->(1,50)
    (1,50)*(50, 100)->(1, 100)
    (1, 100)*(100, 10)->(1,10)
    '''
    accuracy_cnt = 0
    x_dataset = x_test
    y_dataset = t_test
    LEN = len(x_dataset)
    accuracy_cnt_list = []
    for i in range(LEN):
        # print(dataset[i].shape)  #(784,)
        y = predict(network, x_dataset[i])  # (10,)
        p = np.argmax(y)
        if p == y_dataset[i]:
            accuracy_cnt += 1
            accuracy_cnt_list.append(float(accuracy_cnt) / (i + 1))

    print("Accuracy:" + str(float(accuracy_cnt) / LEN))

    x = np.arange(len(accuracy_cnt_list))
    y = np.array(accuracy_cnt_list)
    plt.ylim(np.min(accuracy_cnt_list) - 0.01, np.max(accuracy_cnt_list) + 0.01)
    plt.plot(x, y)
    plt.savefig("result_x_train.jpg")  # Accuracy:0.9357666666666666
    # plt.savefig("result_x_test.jpg")  # Accuracy:0.9352

    plt.show()
