import time
import matplotlib.pyplot as plt
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

all_strokes, strokes_collections, labels = ObjectUtil.xml_to_Strokes('./tst/2_2e6b6882-1ba3-44ce-8779-06663ab97ce7.xml')
t = sum(np.array([s.get_t() for s in all_strokes]), [])
draw_frequency(t)
for s in all_strokes:
    t = s.get_t()
    o = UnlabeledObject([s])
    o.visualize(show=True)
    draw_frequency(t)
    plt.show()