from Stroke import Stroke
from Point import Point
from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from ObjectUtil import ObjectUtil
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from RegisterationUtils import RegsiterationUtils


def main():
    reg = Registration('./test_samples/a2.xml', './test_samples/b2.xml', mn_stroke_len=4, re_sampling=0.5)
    # test_single_obj(reg, 2, 2)
    p = reg.register()
    print(p)
    t = []
    for lst in p:
        t.append(RegsiterationUtils.obtain_transformation_matrix(lst))
    print(t)
    morph = Morph(reg.original_obj, reg.target_obj)
    morph.animate_all(t, save=False, file="example3-seq.mp4")


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')


def test_single_obj(reg, org_ind, tar_ind):
    obj1, obj2 = reg.original_obj[org_ind], reg.target_obj[tar_ind]
    d, t = RegisterTwoObjects(reg.original_obj[org_ind], reg.target_obj[tar_ind], reg.total_cost).optimize(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    print(t)
    d /= len(reg.original_obj[org_ind])
    print(len(reg.original_obj[org_ind]), d)
    q = RegsiterationUtils.obtain_transformation_matrix(t)
    print(q)
    t = RegsiterationUtils.get_seq_translation_matrices(t)
    print("t", t)
    reg.original_obj[org_ind].visualize(show=False)
    trans(obj1, reg, org_ind, t[0])
    trans(obj1, reg, org_ind, t[1])
    trans(obj1, reg, org_ind, t[2])
    trans(obj1, reg, org_ind, t[3])
    trans(obj1, reg, org_ind, t[4])
    #trans(obj1, reg, org_ind, t0)
    reg.original_obj[org_ind].visualize(show=False)
    reg.target_obj[tar_ind].visualize(show=False)
    print("t", t)
    print(reg.original_obj[org_ind].get_x())
    plt.show()

def trans(obj1, reg, org_ind, t):
    reg.original_obj[org_ind].transform(t)

if __name__ == '__main__':
    main()
