import numpy as np
import matplotlib.pylab as plt

from mypackge.ch03 import step_function

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.ylim(-0.1, 1.1)
    plt.plot(x, y)
    plt.show()

