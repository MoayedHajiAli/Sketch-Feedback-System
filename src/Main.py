from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegistrationUtils import RegistrationUtils
import copy
array = np.array
from UnlabeledObject import UnlabeledObject
from Stroke import Stroke

# import sys
# sys.stdout = open('results.txt', 'w')

def main():
    reg = Registration('./test_samples/a7.xml', './test_samples/b7.xml', mn_stroke_len=6, re_sampling=1.0, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        # add missing objects
        add_objects(reg, [])
        # p = reg.register()
        p = [[ 2.61272810e+00,  2.80665102e+00,  1.40671684e-09,
        -7.42421390e-02, -9.61967871e-02,  1.24531506e+03,
         2.57588633e+02],
       [ 2.46507804e+00,  2.49869560e+00, -2.59036878e-02,
        -8.93862373e-09, -6.27860131e-02,  4.98804972e+02,
         3.33768332e+02],
       [ 2.48606838e+00,  2.79560685e+00, -2.30429246e-10,
        -1.02455528e-01, -6.92126905e-09,  6.35468178e+02,
         2.93362099e+02],
       [ 3.16425057e+00,  3.57798657e+00, -8.00930776e-09,
        -1.86098312e-07, -8.08472029e-07,  1.16239580e+03,
         3.31343756e+02],
       [ 2.78896120e+00,  3.04152740e+00, -3.65477184e-09,
        -1.26453594e-01, -3.91518487e-07,  1.39655731e+03,
         3.34479979e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  7.17016643e+02,
        -2.41950072e+02],
       [ 2.82865698e+00,  3.05806181e+00,  1.98919413e-01,
        -2.59705614e-01, -7.26457662e-09,  1.41807860e+03,
         2.37146113e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  6.30536515e+02,
        -3.58926970e+02]]
        print([np.array(p)])
        t = []
        for lst in p:
            t.append(RegistrationUtils.obtain_transformation_matrix(lst))
        print(t)
        morph = Morph(reg.original_obj, reg.target_obj)
        morph.seq_animate_all(p, save=False
                              , file="./test_videos/example6-seq.mp4")


def print_lst(lst):
    st = ','.join(map(str, lst))
    print('[', st, ']')

def add_objects(reg, lst):
    for ind in lst:
        add_obj(reg, reg.target_obj[ind])


def add_obj(reg, obj):
  tmp = copy.deepcopy(obj.get_strokes())
  new_obj = UnlabeledObject(tmp)
  print("origin", obj.origin_x, obj.origin_y)
  print("origin", new_obj.origin_x, new_obj.origin_y)
  obj.print_strokes()
  eps = 0.001
  tmp = []
  for st in new_obj.get_strokes():
      for pt in st.get_points():
          pt.x = pt.x * eps + (1 - eps) * new_obj.origin_x
          pt.y = pt.y * eps + (1 - eps) * new_obj.origin_y
      tmp.append(Stroke(st.get_points()))
  new_obj = UnlabeledObject(tmp)
  reg.original_obj.append(new_obj)

def test_single_obj(reg, org_ind, tar_ind):
    obj1, obj2 = reg.original_obj[org_ind], reg.target_obj[tar_ind]
    i, j = org_ind, tar_ind
    x_dif = reg.target_obj[j].origin_x - reg.original_obj[i].origin_x
    y_dif = reg.target_obj[j].origin_y - reg.original_obj[i].origin_y
    d = RegisterTwoObjects(reg.original_obj[org_ind], reg.target_obj[tar_ind], reg.total_cost).calc_dissimilarity(t)
    t = [ 1.14772552e+00,  9.33270160e-01, -1.48902180e-01,
         6.30043243e-02, -7.14737512e-09,  7.34028622e+02,
         1.13881862e+03]
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
