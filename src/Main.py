from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegisterationUtils import RegsiterationUtils

array = np.array

# import sys
# sys.stdout = open('results.txt', 'w')

def main():
    reg = Registration('./test_samples/a6.xml', './test_samples/b6.xml', mn_stroke_len=6, re_sampling=1.0, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        # p = reg.register()
        print([np.array(p)])
        t = []
        for lst in p:
            t.append(RegsiterationUtils.obtain_transformation_matrix(lst))
        print(t)
        morph = Morph(reg.original_obj, reg.target_obj)
        morph.seq_animate_all(p, save=False
                              , file="./test_videos/example6-seq.mp4")


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')


def test_single_obj(reg, org_ind, tar_ind):
    obj1, obj2 = reg.original_obj[org_ind], reg.target_obj[tar_ind]
    i, j = org_ind, tar_ind
    x_dif = reg.target_obj[j].origin_x - reg.original_obj[i].origin_x
    y_dif = reg.target_obj[j].origin_y - reg.original_obj[i].origin_y
    d, t = RegisterTwoObjects(reg.original_obj[org_ind], reg.target_obj[tar_ind], reg.total_cost).optimize(np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif+80]))
    print(len(reg.original_obj[org_ind]), d)
    # t = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif])
    print([np.array(t)])
    morph = Morph([reg.original_obj[org_ind]], [reg.target_obj[tar_ind]])
    morph.seq_animate_all([t], save=False, file="./test_videos/example5-seq-obj3.mp4")
    plt.show()

def trans(obj1, reg, org_ind, t):
    reg.original_obj[org_ind].transform(t)

if __name__ == '__main__':
    main()
