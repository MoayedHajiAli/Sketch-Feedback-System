from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegisterationUtils import RegsiterationUtils

array = np.array

# import sys
# sys.stdout = open('results.txt', 'w')

def main():
    reg = Registration('./test_samples/a4.xml', './test_samples/b4.xml', mn_stroke_len=4, re_sampling=1.0, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        p = reg.register()
        # p = [[-1.00000001e+00, -1.28788863e+00, -1.69752419e-01,
        #   6.31450838e-07, -1.12070523e-06, 4.54286208e+02,
        #   4.12660537e+02],
        #  [1.25815754e+00, 9.99999982e-01, -1.47724515e-02,
        #   -4.12106332e-09, -1.87179887e-04, 2.98805319e+02,
        #   9.65174921e+02],
        #  [1.12261746e+00, 1.49375754e+00, -7.74722417e-09,
        #   -1.89499735e-09, -6.34406258e-09, 7.46381692e+02,
        #   9.67120240e+02],
        #  [1.87224035e+00, -1.42851767e+00, -8.93409880e-03,
        #   2.79033201e-02, -1.69886231e-09, 7.38741398e+02,
        #   1.00785358e+03],
        #  [9.50351024e-01, 1.86784189e+00, -1.26953590e-08,
        #   -3.84243796e-09, -3.62022997e-03, -4.67436994e+01,
        #   7.55370392e+02]]
        print([np.array(p)])
        t = []
        for lst in p:
            t.append(RegsiterationUtils.obtain_transformation_matrix(lst))
        print(t)
        morph = Morph(reg.original_obj, reg.target_obj)
        morph.seq_animate_all(p, save=False, file="./test_videos/example5-seq.mp4")


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')


def test_single_obj(reg, org_ind, tar_ind):
    obj1, obj2 = reg.original_obj[org_ind], reg.target_obj[tar_ind]
    # d, t = RegisterTwoObjects(reg.original_obj[org_ind], reg.target_obj[tar_ind], reg.total_cost).optimize(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # print([np.array(t)])
    # print(len(reg.original_obj[org_ind]), d)
    t = array([-5.33093643e-01,  1.02299540e+00,  1.58678591e+00, -1.10427516e-01,
       -3.27549100e-01,  3.34016889e+01,  5.57647374e+02])
    morph = Morph([reg.original_obj[org_ind]], [reg.target_obj[tar_ind]])
    morph.seq_animate_all([t], save=False, file="example3-seq-obj2.mp4")
    plt.show()

def trans(obj1, reg, org_ind, t):
    reg.original_obj[org_ind].transform(t)

if __name__ == '__main__':
    main()
