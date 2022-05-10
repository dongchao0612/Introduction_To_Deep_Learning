import numpy as np
from mypackge.ch05 import Affine

if __name__ == '__main__':
    X = np.array([[1, 2]])
    W = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 1]])
    affine = Affine(W,b)
    result = affine.forward(X)
    affine.backward(dout=1)
    print(affine.dw,affine.db)
