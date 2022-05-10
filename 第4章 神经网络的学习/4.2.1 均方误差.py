import numpy as np

from mypackge.ch04 import mean_squared_error

if __name__ == '__main__':
    # 正确
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    result = mean_squared_error(np.array(t), np.array(y))
    print(result)
    # 错误
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    result1 = mean_squared_error(np.array(t), np.array(y))
    print(result1)
