from scipy.interpolate import interp1d
import numpy as np

x = [1, 2, 3]
y = [4, 5, 6]
x = np.reshape(x, (3, 1))
y = np.reshape(y, (3, 1))
print(x)
print(y)
X = np.concatenate((x, y), axis=1)
print(X)


