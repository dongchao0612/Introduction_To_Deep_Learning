import numpy as np

from mypackge.ch03 import sigmoid, identity_func

if __name__ == '__main__':
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    print("Z1 = ",Z1)

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    print("Z2 = ",Z2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Z3 = identity_func(A3)
    print("Z3 = ",Z3)