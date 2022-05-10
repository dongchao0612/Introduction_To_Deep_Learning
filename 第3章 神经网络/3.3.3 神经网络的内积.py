import numpy as np

if __name__ == '__main__':
    X = np.array([[1, 2]])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    print(X.shape, W.shape)
    y = np.dot(X, W)
    print(y)

