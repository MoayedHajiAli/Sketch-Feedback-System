from scipy.interpolate import interp1d
import numpy as np
from Nearest_search import Nearest_search

x = [1, 2, 3]
y = [4, 5, 6]
tree = Nearest_search(x, y)
print(tree.query([2.1, 1], [5.1, 3]))

