from Morph import Morph
from Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from RegistrationUtils import RegistrationUtils
from ObjectUtil import ObjectUtil
import copy
array = np.array
from UnlabeledObject import UnlabeledObject
from Stroke import Stroke
from Evaluation import Evaluation
import time

def main():
    q = int(input())
    if q == 0:
        evalute()
    elif q == 1:
        reg = Registration('./test_samples/a8.xml', './test_samples/b8.xml', mn_stroke_len=6, re_sampling=0.5, flip=True, shift_target_y = 1000)
        a, b = map(int, input().split())
        test_single_obj(reg, a, b)
    else:
        reg = Registration('./test_samples/a8.xml', './test_samples/b8.xml', mn_stroke_len=6, re_sampling=0.5, flip=True, shift_target_y = 1000)
        a, b = map(int, input().split())
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
    st = time.time()
    obj1, obj2 = reg.original_obj[i], reg.target_obj[j]
    obj1 = ObjectUtil.object_restructure(obj1, n = 30)
    x_dif = obj2.origin_x - obj1.origin_x
    y_dif = obj2.origin_y - obj1.origin_y
    p = array([ 3.05030249e+00,  3.45396355e+00,  7.05516539e-02, -2.43930107e-05,
       -2.67216045e-10,  1.16495973e+03,  3.21120536e+02])
    d, p = RegisterTwoObjects(obj1, obj2, reg.total_cost).optimize()
    print(d, [np.array(p)])
    print(RegistrationUtils.identify_similarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p)))
    print("Running time: ", time.time()-st)
    morph = Morph([obj1], [obj2])
    print("original len", len(reg.original_obj[i]))
    print("target len", len(reg.target_obj[j]))
    morph.seq_animate_all([p], save=False, file="./test_videos/example7-obj3-4.mp4")
    plt.show()

def evalute():
    eval = Evaluation([], [], re_sampling=0.5)
    eval.add_file('prototypes/p1.xml')
    eval.add_file('prototypes/p2.xml')
    eval.add_file('prototypes/p3.xml')
    eval.add_file('prototypes/p4.xml')
    print("Labels: ", eval.labels)
    # ../ASIST_Dataset/Data/Data_A
    eval.start('../ASIST_Dataset/Data/Data_A', 100)
    # print("Prediction Accuracy is: ", acc)
    # print("Confusion matrix:")

if __name__ == '__main__':
    main()
