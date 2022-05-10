import numpy as np

from mypackge.ch03 import sigmoid


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_mini_onehot(y, t):
    delta = 1e-7
    if y.ndim == 1:
        # print("y.ndim=",y.ndim)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # print("batch_size=",batch_size)
    return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error_mini_labels(y, t):
    delta = 1e-7
    if y.ndim == 1:
        # print("y.ndim=",y.ndim)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # print("shape of y: ", y.shape)
    # print("shape of t: ", t.shape)
    batch_size = y.shape[0]
    # print("batch_size=",batch_size)
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return np.sum(x ** 2)


def _numerical_gradient_1d(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #print(it)
    while not it.finished:
        #print("\t\t\tnumerical_gradient")
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        #print(grad)
        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient_2d(f, x)
        x -= lr * grad
    return x





