import numpy as np

from mypackge.ch04 import cross_entropy_error_mini_onehot, cross_entropy_error_mini_labels

if __name__ == '__main__':
    y = np.array([[0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.2, 0.8, 0, 0, 0, 0, 0, 0]])
    t = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])  # one-hot

    result = cross_entropy_error_mini_onehot(y, t)
    print(result)

    y = np.array([[0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0.2, 0.8, 0, 0, 0, 0, 0, 0]])
    t_onehot = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])  # one-hot
    t = t_onehot.argmax(axis=1)  # 非one-hot
    print(t)
    result1 = cross_entropy_error_mini_labels(y, t)
    print(result1)

    # for i in range(2):
    #     for j in range(10):
    #         print(cross_entropy_error_mini_labels(y, [i, j]), [i, j])
