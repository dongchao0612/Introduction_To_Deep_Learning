import numpy as np

from mypackge.ch04 import cross_entropy_error

if __name__ == '__main__':
    # 正确
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    result = cross_entropy_error(np.array(y), np.array(t))
    print(result)
    # 错误
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    result1 = cross_entropy_error(np.array(y), np.array(t))
    print(result1)
