# import matplotlib
# matplotlib.use('TkAgg')
import time
import numpy as np

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

with open("log/analysis.txt") as f:
    arr = list(map(float, f.read().split('\n')[:-1]))

# arr = np.array(arr, dtype=float)
A, B = arr[:128], arr[128:]

A = sorted(A)
B = sorted(B)


for i in range(len(A)):
    print(A[i]*1000, B[i]*1000, A[i]-B[i])

print(sum(A), sum(B))


# square - parallelogram right: 18.3
# Square - Square: 13.2
# Trap up - Trap down : 23.6
# Trap up - square: 26.8
# squarre - diamond: 26.8
# x - y : 28.4

# with a larger number of points, the embeddings are better.