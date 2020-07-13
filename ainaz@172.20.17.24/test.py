from scipy.interpolate import interp1d
import numpy as np


x = np.array([0, 1, 2, 3, 2.1, 0.5])
y = [1, 2, 3, 4, 5, 6]
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
xnew = np.array([0, 0.9, 2, 2.5, 2.1, 0.5])
import matplotlib.pyplot as plt
plt.plot(x, y, '-', xnew, f(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()