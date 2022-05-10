import numpy as np

from mypackge.ch05 import SoftmaxWithLoss

if __name__ == '__main__':
    swl = SoftmaxWithLoss()
    x = np.array([1, 2, 3])
    t = np.array([0, 1, 0])
    result = swl.forward(x, t)
    print(result)
