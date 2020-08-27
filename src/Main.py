from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegistrationUtils import RegistrationUtils
import copy
array = np.array
from UnlabeledObject import UnlabeledObject
from Stroke import Stroke

def main():
    reg = Registration('./test_samples/a7.xml', './test_samples/b7.xml', mn_stroke_len=6, re_sampling=1.0, flip=True, shift_target_y = 1000)
    a, b = map(int, input().split())
    if a != -1 and b != -1:
        test_single_obj(reg, a, b)
    else:
        # add missing objects (temporarily as training is running on ssh server)
        add_objects(reg, [])
        p = reg.register()
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

def test_single_obj(reg, i, j):

    obj1, obj2 = reg.original_obj[i], reg.target_obj[j]
    x_dif = reg.target_obj[j].origin_x - reg.original_obj[i].origin_x
    y_dif = reg.target_obj[j].origin_y - reg.original_obj[i].origin_y
    # t = [ 1.14772552e+00,  9.33270160e-01, -1.48902180e-01,
    #      6.30043243e-02, -7.14737512e-09,  7.34028622e+02,
    #      1.13881862e+03]
    # t = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x, 0.0])
    d, p = RegisterTwoObjects(reg.original_obj[i], reg.target_obj[j], reg.total_cost).optimize()

    print([np.array(p)])
    morph = Morph([reg.original_obj[i]], [reg.target_obj[j]])
    print("original len", len(reg.original_obj[i]))
    print("target len", len(reg.target_obj[j]))
    morph.seq_animate_all([p], save=False, file="./test_videos/example5-seq-obj3.mp4")
    plt.show()

def trans(obj1, reg, org_ind, t):
    reg.original_obj[org_ind].transform(t)

if __name__ == '__main__':
    main()
