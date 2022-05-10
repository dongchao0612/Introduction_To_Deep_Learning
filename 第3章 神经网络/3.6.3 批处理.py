import pickle
import matplotlib.pylab as plt
import numpy as np

from mypackge.ch03 import softmax, sigmoid
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
    # print("Z1.shape = ", z1.shape)#Z1.shape =  (100, 50)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    # print("Z2.shape = ", z2.shape)#Z2.shape =  (100, 100)

    a3 = np.dot(z2, w3) + b3
    z3 = softmax(a3)
    # print("Z3.shape = ", z3.shape)#Z3.shape =  (100, 10)

    return z3


if __name__ == '__main__':

    x_train, t_train, x_test, t_test = get_data()
    network = init_network()

    '''
    (100,784)*(784, 50)->(100, 50)
    (100,50)*(50, 100)->(100, 100)
    (100, 100)*(100, 10)->(100, 10)
    '''

    batch_size = 100
    accuracy_cnt = 0
    x_dataset = x_test
    y_dataset = t_test
    LEN = len(x_dataset)
    accuracy_cnt_list = []
    for i in range(0, LEN, batch_size):
        x_batch = x_dataset[i:i + batch_size]
        # print(x_batch.shape)#(100, 784)
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        #print(p.shape)  # (100,)
        accuracy_cnt += np.sum(p == y_dataset[i:i + batch_size])
        # print("accuracy_cnt=",accuracy_cnt,"i=",(i+batch_size))
        accuracy_cnt_list.append(accuracy_cnt / (i + batch_size))

    print("Accuracy:"+str(float(accuracy_cnt)/LEN)) # Accuracy:0.9357666666666666

    x = np.arange(len(accuracy_cnt_list))
    y = np.array(accuracy_cnt_list)
    plt.ylim(np.min(accuracy_cnt_list) - 0.01, np.max(accuracy_cnt_list) + 0.01)
    plt.plot(x, y)
    #plt.savefig('result_x_train_batch.jpg')#Accuracy:0.9357666666666666
    plt.savefig('result_x_test_batch.jpg')
    plt.show()
