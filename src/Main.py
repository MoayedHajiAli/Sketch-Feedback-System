from animator.SketchAnimation import SketchAnimation
from register.Registration import Registration, RegisterTwoObjects
from matplotlib import pyplot as plt
import numpy as np
from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
import copy
from sketch_object.UnlabeledObject import UnlabeledObject
from sketch_object.Stroke import Stroke
from tools.ClassEvaluation import ClassEvaluation
from tools.ObjectParsing import ObjectParsing
from tools.StrokeClustering import DensityClustering
from tools.ParsingEvaluation import ParsingEvaluation
import time

array = np.array

"""Summary from last time

1- sketchformer is working good with classification, and its embeddings are a good indication of how similar two 
    sketches are. However, optimizing the transformation parameters takes so much time (around 60 seconds) and not always working.
    The optimization is highly dependent on the learning rete and eps. Therefore, it is hard to use it for optimization.

"""

def main():
    q, s = map(int, input().split())
    if q == 0:
        evalute()
    elif q == 1:
        reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)
        a, b = map(int, input().split())

        # initial transformation test
        # reg.original_obj[a].transform(RegistrationUtils.obtain_transformation_matrix(np.array([0.4, 0.6, 1.5, 1.0, 1.2, 0.0, 20.0])))
        # reg.original_obj[a].transform(RegistrationUtils.obtain_transformation_matrix(np.array([2, 5, 0, 2, 1.2, 0.3, 20.0])))
        # reg.original_obj[a] = reg.original_obj[a].get_copy()
        # reg.original_obj[a].reset()
        test_single_obj(reg, a, b)

    elif q == 2:
        reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=0.5, flip=False, shift_target_y = 0)
        # add missing objects (temporarily as pre-calculation is running on ssh server)
        add_objects(reg, [])
        st = time.time()
        # p = reg.register(mx_dissimilarity=100)
        p = array([[ 1.45722464e+00,  7.78708354e-01, -9.44946492e-02, 
         3.73263963e-05, -1.15940180e-01,  1.30174690e+02,
         1.08618391e+03],
       [ 2.99213538e+00,  2.13993443e+00, -3.12549236e-01,
        -9.37196010e-09, -2.51166923e-01,  8.40984305e+01,
         1.16747442e+03],
       [ 1.95457770e+00,  1.42807220e+00,  8.39620404e-02,
        -1.06922787e-01, -4.11040840e-05,  2.08639959e+01,
         1.25252956e+03],
       [ 1.43430632e+00,  1.07209142e+00, -6.62935444e-03,
        -2.30945611e-07, -5.39165195e-01,  7.71330284e+01,
         1.10928334e+03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  4.62000000e+02,
        -4.05000000e+02],
       [ 1.18456758e+00,  1.27550794e+00, -7.20732174e-03,
         2.58774376e-03, -2.13672718e-08,  7.53921708e+01,
         1.09814690e+03],
       [ 2.58470012e+00,  9.56816447e-01, -3.70635071e-01,
        -2.72496980e-06, -4.20030789e-01,  1.78376036e+02,
         1.15777014e+03],
       [ 1.44212861e+00,  1.17999648e+00,  1.64021099e-01,
        -8.79004499e-10,  3.92330318e-07,  1.42286550e+02,
         1.19926614e+03],
       [ 1.81472972e+00,  1.59928339e+00, -1.37038069e-10,
        -1.28855308e-08, -1.61823148e-02,  1.54550722e+02,
         1.19518243e+03],
       [ 1.63076665e+00,  1.07393475e+00, -4.38055197e-01,
        -7.10354326e-09, -2.64654568e-02,  1.35055690e+02,
         1.21605584e+03],
       [ 1.54305145e+00,  9.69748552e-01, -1.24442692e-06,
        -1.11351752e-01, -8.94998428e-01,  1.89348773e+02,
         1.15696874e+03],
       [ 1.64911161e+00,  1.23393224e+00, -1.42400271e-01,
        -9.26788831e-10, -1.83333449e-08,  1.53865573e+02,
         1.19684491e+03],
       [ 2.86416903e+00,  2.61255450e+00,  2.30052716e-01,
        -2.29782984e-01,  6.63516741e-08, -1.38118874e+02,
         1.11772507e+03],
       [ 2.49562560e+00,  1.68228166e+00,  1.21750135e-04,
        -2.24783616e-01,  7.54700560e-05,  5.17260916e+01,
         1.32091431e+03],
       [ 1.64681651e+00,  1.52351453e+00, -7.83647669e-09,
         1.60794023e-02,  1.61237400e-01, -5.71557289e+01,
         1.18280595e+03],
       [ 1.36926448e+00,  1.01539244e+00,  3.89119541e-01,
        -8.32254598e-02, -2.06880516e-05,  1.43205387e+02,
         1.08491875e+03],
       [ 3.10879357e+00,  2.65746324e+00, -2.98185659e-01,
         1.11458847e-08, -1.21935039e-06,  8.99983892e+01,
         1.15497242e+03],
       [ 1.00000000e+03,  1.00000000e+03,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00]])
        print([np.array(p)])
        t = []
        for lst in p:
            t.append(RegistrationUtils.obtain_transformation_matrix(lst))
        print(t)
        print("Running time:", time.time()-st)
        SketchAnimation = SketchAnimation(reg.original_obj, reg.target_obj)
        SketchAnimation.seq_animate_all(p, save=False
                              , file="./test_videos/example6-seq.mp4")
    elif q == 3:
        # find correspondences of an object
        # reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=0.3, flip=True, shift_target_y = 0)
        find_correspondences('../ASIST_Dataset/Data/Data_A/MoneyQuestion', '../ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml')

    elif q == 4:
        # find the embeddings
        reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=1, flip=True, shift_target_y = 0)
        embds = ObjectUtil.get_embedding(np.concatenate([reg.original_obj, reg.target_obj]))
        org_embd = embds[:len(reg.original_obj)]
        tar_embd = embds[len(reg.original_obj):]

        for i, embd1 in enumerate(org_embd):
          for j, embd2 in enumerate(tar_embd):
            print(reg.origninal_labels[i], reg.target_labels[j], np.linalg.norm(embd1 - embd2))

        # for i in range(len(embd1)):
        #   print(float(embd1[i]), float(embd2[i]), float(embd3[i]))
        print("The predicted classes of the objects are",  ObjectUtil.classify(np.concatenate((reg.original_obj, reg.target_obj))))
    
    elif q == 5:
      #evaluate parsing
      evaluator = ParsingEvaluation('../ASIST_Dataset/Data/Data_A/MoneyQuestion', '../ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml', n_files=100)
      evaluator.evaluate()

    elif q == 6:
      tmp_test('./input_directory/samples/test_samples/a' + str(s) + '.xml')

    elif q == 7:
      # go from strokes to object by comparing the embeddings of all combinations of strokes
      org_strokes_lst = ObjectUtil.xml_to_strokes('./input_directory/samples/test_samples/a' + str(s) + '.xml', re_sampling=1.0, flip=True)
      tar_strokes_lst = ObjectUtil.xml_to_strokes('./input_directory/samples/test_samples/b' + str(s) + '.xml', re_sampling=1.0, flip=True)

      # UnlabeledObject(org_strokes_lst).visualize()
      # UnlabeledObject(tar_strokes_lst).visualize()

      # test
      # tmp = []
      # for i in range(0, 7):
      #   tmp.append(org_strokes_lst[i])
      # obj1 = UnlabeledObject(tmp)

      # tmp = []
      # for i in range(3, 5):
      #   tmp.append(tar_strokes_lst[i])
      # obj2 = UnlabeledObject(tmp)

      # obj1.visualize()
      # obj2.visualize()
      # print(len(obj1), len(obj2))
        
      # embd1, embd2, embd3 = ObjectUtil.get_embedding([obj1, obj2, obj2])

      # for i in range(len(embd1)):
      #   print(float(embd1[i]), float(embd2[i]), float(embd3[i]))

      # print("The norm of the difference vertor between the embeddings", np.linalg.norm(embd1 - embd2)) 
      # return      

      # prepare objects of all combinations of strokes
      org_obj, tar_obj = [], []
      for i in range(len(org_strokes_lst)):
        tmp= []
        total_len = 0
        for j in range(i, len(org_strokes_lst)):
          tmp.append(org_strokes_lst[j].get_copy())
          total_len += len(org_strokes_lst[j])
          if total_len > 200:
            break
          org_obj.append([UnlabeledObject(copy.deepcopy(tmp)), [i, j]])
        
      for i in range(len(tar_strokes_lst)):
        tmp = []
        total_len = 0
        for j in range(i, len(tar_strokes_lst)):
          tmp.append(tar_strokes_lst[j].get_copy())
          total_len += len(tar_strokes_lst[j])
          if total_len > 200:
            break
          tar_obj.append([UnlabeledObject(copy.deepcopy(tmp)), [i, j]])

      org_obj = sorted(org_obj, key=lambda a: a[1][1] - a[1][0])

      # obtain embeddings
      lst1, lst2 = [a[0] for a in org_obj], [a[0] for a in tar_obj]
      embds = ObjectUtil.get_embedding(np.concatenate([lst1, lst2]))
      org_embd, tar_embd = embds[:len(lst1)], embds[len(lst1):]

      org_vis, tar_vis = {}, {}
      org_res, tar_res = [], []
      t = 15
      
      for i, (embd1, obj1) in enumerate(zip(reversed(org_embd), reversed(org_obj))): 
        for j, (embd2, obj2) in enumerate(zip(reversed(tar_embd), reversed(tar_obj))):
          d = np.linalg.norm(embd2 - embd1)
          if d < 18:
            print(np.linalg.norm(embd2 - embd1))
            obj1[0].visualize()
            obj2[0].visualize()
            for a, b in zip(embd1, embd2):
              print(float(a), float(b), float(b-a))

      return -1
      for i, (embd1, obj1) in enumerate(zip(reversed(org_embd), reversed(org_obj))):
        brk_process = False
        l, r = obj1[1][0], obj1[1][1]
        for i in range(l, r+1):
          if i in org_vis:
            brk_process = True
            break
        if brk_process:
          continue

        for j, (embd2, obj2) in enumerate(zip(reversed(tar_embd), reversed(tar_obj))):
          brk_process = False
          l, r = obj2[1][0], obj2[1][1]
          for i in range(l, r+1):
            if i in tar_vis:
              brk_process = True
              break
          if brk_process:
            continue
          
          l, r = obj1[1][0], obj1[1][1]
          print("org", l, r)
          l, r = obj2[1][0], obj2[1][1]
          print("tar", l, r)
          print(np.linalg.norm(embd2 - embd1))

          if np.linalg.norm(embd2 - embd1) < t:
            a1, a2 = ObjectUtil.get_embedding([obj1[0], obj2[0]])
            print("The norm of the difference vertor between the embeddings", np.linalg.norm(a1 - a2))  

            org_res.append(obj1[0])
            tar_res.append(obj2[0])

            obj1[0].visualize()
            obj2[0].visualize()

            l, r = obj1[1][0], obj1[1][1]
            print("org", l, r)
            for i in range(l, r+1):
              org_vis[i] = True
            l, r = obj2[1][0], obj2[1][1]
            print("tar", l, r)
            for i in range(l, r+1):
              tar_vis[i] = True
            

# Sketchformer summary:     
  # sketchformer is not variant to translation, or diagonal scaling
  # sketchformer is variant to rotation, or shearing 
  # skethcformer does not change at all for very small step sizes (1e6 or more)
  # after ceratin norm differenc between two sketches (around 30), it becomes harder to make it larger (even with extreem trans) ???
  # F(A + B) != F(A) + F(B) // for embeddings
  # if the objects are not the same, the embedding will not tell us much about how different they are.
  #       For example, parallelogram and trapazoid up -> 21.5, trapazoid up and trapazoid down -> 23.5


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
    print("objects length", len(obj1), len(obj2))

    embd1, embd2 = ObjectUtil.get_embedding([obj1, obj2])
    print("The norm of the difference vertor between the embeddings before registration", np.linalg.norm(embd1 - embd2))

    d, p = RegisterTwoObjects(obj1, obj2, reg.total_cost).optimize()
    print(d, [np.array(p)])
    print("Running time: ", time.time()-st)
    # print(RegistrationUtils.identify_similarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p)))
    animation = SketchAnimation([obj1], [obj2])
    # print(RegistrationUtils.calc_dissimilarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p), target_dis=False))
    animation.seq_animate_all([p], save=False, file="./test_videos/example7-obj3-4.mp4")
    plt.show()

    obj1.reset()
    obj1.transform(RegistrationUtils.obtain_transformation_matrix(p))

    cls1, cls2 = ObjectUtil.classify([obj1, obj2])
    print("")
    print("The classifications of the objects are", cls1, cls2)


    embd1, embd2 = ObjectUtil.get_embedding([obj1, obj2])
    # print(len(embd1), len(embd2))
    # for i in range(len(embd1)):
    #   print(abs(float(embd1[i]) - float(embd2[i])))
    print("The norm of the difference vertor between the embeddings after registration", np.linalg.norm(embd1 - embd2))


def find_correspondences(dir, obj):
  cluster = DensityClustering.fromDir(obj, dir, n=2)
  cluster.reg_based_mut_execlusive_cluster()

def evalute():
    eval = ClassEvaluation([], [], re_sampling=0.5)
    eval.add_file('prototypes/p1.xml')
    eval.add_file('prototypes/p2.xml')
    eval.add_file('prototypes/p3.xml')
    eval.add_file('prototypes/p4.xml')
    print("Labels: ", eval.labels)
    # ../ASIST_Dataset/Data/Data_A
    eval.start('../ASIST_Dataset/Data/Data_A', 100)
    # print("Prediction Accuracy is: ", acc)
    # print("Confusion matrix:")

def tmp_test(file):
    stroke_lst = ObjectUtil.xml_to_strokes(file)
    tmp = []
    for st in stroke_lst:
      tmp.append(st)
      obj = UnlabeledObject(tmp)
      obj.visualize()


if __name__ == '__main__':
    main()
