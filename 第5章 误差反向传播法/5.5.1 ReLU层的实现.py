import numpy as np

from mypackge.ch05 import Relu

if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    rule=Relu()
    result=rule.forward(x)
    print(result)
    dresult=rule.backward(np.array([[2.0,1.0],[1.0,1.0]]))
    print(dresult)


