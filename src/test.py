# import matplotlib
# matplotlib.use('TkAgg')
import time
import numpy as np
from ObjectUtil import ObjectUtil
from UnlabeledObject import UnlabeledObject

def draw_frequency(t):
    diff = []
    for i in range(len(t) - 1):
        diff.append(t[i + 1] - t[i])
    x = np.array([i for i in range(len(diff))])
    plt.plot(x, diff)
    plt.show()

# all_strokes, strokes_collections, labels = ObjectUtil.xml_to_Strokes('./tst/2_2e6b6882-1ba3-44ce-8779-06663ab97ce7.xml')
# t = sum(np.array([s.get_t() for s in all_strokes]), [])
# draw_frequency(t)
# for s in all_strokes:
#     t = s.get_t()
#     o = UnlabeledObject([s])
#     o.visualize(show=True)
#     draw_frequency(t)
#     plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fig= plt.figure()
ax= fig.add_subplot(111)
ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color= "lightblue", linewidth= 3)
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color= "darkgreen", marker= "^")
ax.set_xlim(0.5, 4.5)
plt.show()