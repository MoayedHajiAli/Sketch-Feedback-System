from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegisterationUtils import RegsiterationUtils
import copy
array = np.array
from UnlabeledObject import UnlabeledObject
from Stroke import Stroke

# import sys
# sys.stdout = open('results.txt', 'w')

def main():
    reg = Registration('./test_samples/a5.xml', './test_samples/b4.xml', mn_stroke_len=6, re_sampling=1.0, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        # add missing objects
        add_obj(reg, reg.target_obj[3])
        add_obj(reg, reg.target_obj[2])
        # p = reg.register()
        p = [[ 9.99999618e-01,  8.67910670e-01, -1.61348840e-01,
        -7.48704891e-10,  4.86081810e-07,  2.13155474e+02,
         9.57682057e+02],
       [ 1.00735391e+00,  9.99999989e-01, -3.84887913e-02,
        -1.95178502e-09, -7.28796116e-08,  1.43759252e+02,
         8.87648048e+02],
       [ 1.15910153e+00,  8.96285942e-01, -1.84709959e-01,
         2.35241702e-02, -5.81943246e-10,  7.33532999e+02,
         1.03772778e+03],
       [ 1.33156979e+00,  1.17358312e+00,  1.32067401e+00,
        -6.18955158e-09,  5.14391489e-03,  3.34658053e+02,
         7.86647592e+02],
       [ 6.14374676e-01,  1.48277551e+00, -1.28539648e-10,
        -1.34846179e-02, -6.79510281e-02,  2.63227503e+02,
         9.68753503e+02],
       [ 1.00000000e+03,  1.00000000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 1.00000000e+03,  1.00000000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00]]
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
