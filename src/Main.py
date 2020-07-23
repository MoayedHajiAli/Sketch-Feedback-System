from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegisterationUtils import RegsiterationUtils

array = np.array

# import sys
# sys.stdout = open('results.txt', 'w')

def main():
    reg = Registration('./test_samples/a4.xml', './test_samples/b4.xml', mn_stroke_len=4, re_sampling=0.5, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        # p = reg.register()
        p = [[ 1.01049440e+00,  9.71895748e-01, -2.55353196e-02,
        -5.00257524e-09, -2.40657639e-06,  2.25546066e+01,
         9.11585938e+02],
       [ 8.24319572e-01,  8.56894458e-01, -5.96849955e-03,
         4.15145123e-08,  2.47234290e-10, -1.33946786e+01,
         1.01065715e+03],
       [ 1.15680505e+00,  1.29259509e+00,  8.53470471e-02,
        -3.22740300e-07, -2.13792893e-08, -1.31558136e+02,
         9.06720368e+02],
       [ 9.91147142e-01,  8.21106549e-01, -9.99346020e-09,
        -1.03305373e-08, -2.09306092e-09,  1.40735694e+01,
         5.64907224e+02],
       [ 7.36101608e-01,  7.74184577e-01, -4.92438191e-09,
        -3.68027886e-01, -3.08126609e-02,  3.67767322e+02,
         1.04390950e+03],
       [ 1.10047000e+00,  1.63904051e+00,  4.44146567e-02,
        -3.33387140e-02, -1.06906138e-08,  5.54971915e+01,
         9.21884591e+02],
       [ 9.36940387e-01,  6.64989239e-01, -2.07135393e-09,
        -1.84137426e-01, -5.06033056e-02,  2.05018341e+02,
         1.05442516e+03]]
        print([np.array(p)])
        t = []
        for lst in p:
            t.append(RegsiterationUtils.obtain_transformation_matrix(lst))
        print(t)
        morph = Morph(reg.original_obj, reg.target_obj)
        morph.seq_animate_all(p, save=True
                              , file="./test_videos/example4-seq.mp4")


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')


def test_single_obj(reg, org_ind, tar_ind):
    obj1, obj2 = reg.original_obj[org_ind], reg.target_obj[tar_ind]
    i, j = org_ind, tar_ind
    x_dif = reg.target_obj[j].origin_x - reg.original_obj[i].origin_x
    y_dif = reg.target_obj[j].origin_y - reg.original_obj[i].origin_y
    d, t = RegisterTwoObjects(reg.original_obj[org_ind], reg.target_obj[tar_ind], reg.total_cost).optimize(np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif]))
    print(len(reg.original_obj[org_ind]), d)
    # t = array([1.28221351e+00, 1.73830158e+00, 1.00440837e-01, 4.65917198e-07,
    #        1.45163205e-07, 7.60385866e+01, 6.17645329e+02])
    print([np.array(t)])
    morph = Morph([reg.original_obj[org_ind]], [reg.target_obj[tar_ind]])
    morph.seq_animate_all([t], save=False, file="./test_videos/example5-seq-obj3.mp4")
    plt.show()

def trans(obj1, reg, org_ind, t):
    reg.original_obj[org_ind].transform(t)

if __name__ == '__main__':
    main()
