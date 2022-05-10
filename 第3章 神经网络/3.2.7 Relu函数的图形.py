import numpy as np
import matplotlib.pylab as plt

from mypackge.ch03 import relu

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.ylim(-0.1, 1.1)
    plt.plot(x, y)
    plt.show()