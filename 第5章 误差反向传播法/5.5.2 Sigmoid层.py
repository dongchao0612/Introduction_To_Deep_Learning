import numpy as np

from mypackge.ch05 import Sigmoid

if __name__ == '__main__':
    sigmoid = Sigmoid()
    result = sigmoid.forward(np.array([1, 5]))
    print(result)
    dresult = sigmoid.backword(np.array([1]))
    print(dresult)
