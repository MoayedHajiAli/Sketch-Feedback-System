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
from registrationNN.models import NNModel
import time

array = np.array

"""Summary from last time

1- sketchformer is working good with classification, and its embeddings are a good indication of how similar two 
    sketches are. However, optimizing the transformation parameters takes so much time (around 60 seconds) and not always working.
    The optimization is highly dependent on the learning rete and eps. Therefore, it is hard to use it for optimization.

"""

def main():
    elif q == 5:
      #evaluate sketch object level segmentation
      evaluator = ParsingEvaluation('../ASIST_Dataset/Data/Data_A/MoneyQuestion', '../ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml', n_files=5)
      evaluator.evaluate()

    elif q == 6:
      tmp_test('./input_directory/samples/test_samples/a' + str(s) + '.xml')

    elif q == 8:
      # perform a quick test (for now: convertion between stroke-3 and poly format)
      reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=1, flip=True, shift_target_y = 0)
      quick_test(reg.original_obj)

    elif q == 9:
      # go from z̄strokes to object by comparing the embeddings of all combinations of strokes
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
  # TODO: VERY IMPROTANT sketchformer normalize the whole object, however I am normalizing each stroke
  # which makes strokes to be on top of each others
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
  cluster.mut_execlusive_cluster()

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

def quick_test(objs):
  # test if format conversion is working
  obj = objs[1]
  obj.visualize()
  obj = ObjectUtil.poly_to_stroke3([obj])[0]
  print(np.shape(obj))
  obj = np.asarray(obj)
  obj = ObjectUtil.stroke3_to_poly([obj])[0]
  print(type(obj))
  obj.visualize()

if __name__ == '__main__':
    main()
