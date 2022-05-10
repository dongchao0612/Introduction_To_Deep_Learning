import numpy as np

from mypackge.ch04 import function_2, _numerical_gradient_1d, gradient_descent

if __name__ == '__main__':
    x = np.array([3.0, 4.0])
    f = function_2
    result = _numerical_gradient_1d(f, x)
    print(result)
    result1 = gradient_descent(function_2, np.array([-3.0, 4.0]), lr=0.1, step_num=100)
    print(result1)
