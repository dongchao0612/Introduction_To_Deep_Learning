import numpy as np
import matplotlib.pylab as plt

from mypackge.ch04 import numerical_diff, function_1, function_2

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    # plt.show()
    result1 = numerical_diff(function_1, 5)
    result2 = numerical_diff(function_1, 10)
    result3 = function_2(np.array([2, 2]))
    print(result1, result2)
    print(result3)
