from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
from tools.StrokeClustering import DensityClustering, DBScanClustering
from sketch_object.UnlabeledObject import UnlabeledObject
import numpy as np 
from progress.bar import Bar
from tqdm import tqdm
import os
import pathlib
from multiprocessing import Pool
import multiprocessing
import time
import pandas as pd

pd.set_option('max_columns', None)

class ParsingEvaluation:

    def __init__(self, test_dir, tar_file=None, n_files=-1, re_sampling=0.0):
        self.test_dir = test_dir
        self.tar_file = tar_file

        # obtain all objects with their indices and labels
        if self.tar_file:
            self.tar_objs, self.tar_lbs, self.tar_segs = ObjectUtil.xml_to_IndexedUnlabeledObjects(tar_file, re_sampling=re_sampling)
            print("[ParsingEvaluation] taget object contains", len(self.tar_objs), "objects")
        self.test_objs, self.test_lbs, self.test_segs = self.explore(test_dir, n_files=n_files)
        print("[ParsingEvaluation] testing directory contains", sum([len(a) for a in self.test_objs]), "objects")
        self.n_files = n_files

        # remove unwanted objects
        self.clean()

        if self.tar_file:
            print("[ParsingEvaluation] after cleaning, taget object contains", len(self.tar_objs), "objects")
        print("[ParsingEvaluation] after cleaning, testing directory contains", sum([len(a) for a in self.test_objs]), "objects")

    def explore(self, directory, n_files=-1, re_sampling=0.0):
        # explore n_files in the directory. extracts the objects and their stroke's ordering 
        unorderd_objs = 0

        # objs : (N x M) M ordered objects for N sketches
        # labels: (N x M) for each object m in sketch n, store its label
        objs, labels, segs = [], [], []

        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                self.explore(path)
            elif path.is_file():
                if n_files != -1 and len(objs) > n_files:
                    break
                if str(path).endswith(".xml"): 
                    try:
                        # extract all objects in the file along with their labels, and indices
                        res_obj, res_lbl, res_segs = ObjectUtil.xml_to_IndexedUnlabeledObjects(str(path), re_sampling=re_sampling)
                        
                        # discard any unordered objects
                        tmp_obj, tmp_lbl, tmp_segs = [], [], []
                        for obj, lbl, seg in zip(res_obj, res_lbl, res_segs):
                            if seg != [*range(seg[0], seg[-1]+1)]:
                                unorderd_objs += 1
                            else:
                                tmp_obj.append(obj)
                                tmp_lbl.append(lbl)
                                tmp_segs.append((seg[0], seg[-1]))
                        
                        # add all resulted objects
                        if len(tmp_obj) > 0:
                            objs.append(tmp_obj)
                            labels.append(tmp_lbl)
                            segs.append(tmp_segs)
                    except Exception as e: 
                        print("ParsingEvaluation] error: could not read file succefully " + str(path))
                        print(str(e))

        print("[ParsingEvaluation] warn:", unorderd_objs, "unordered objects were found and discarded" )
        return objs, labels, segs

    def clean(self, org_removable_labels = ['Number'], tar_removable_labels = ['Number']):
        """Clean the target and orignal storkes by removing unrelated objects such as handwritting
        """
        if self.tar_file:
            i = 0
            while i < len(self.tar_objs):
                if self.tar_lbs[i] in tar_removable_labels:
                    self.tar_lbs.pop(i)
                    self.tar_objs.pop(i)
                    self.tar_segs.pop(i)
                else:
                    i += 1
        
        for i in range(len(self.test_objs)):
            j = 0
            while j < len(self.test_objs[i]):
                if self.test_lbs[i][j] in org_removable_labels:
                    self.test_lbs[i].pop(j)
                    self.test_objs[i].pop(j)
                    self.test_segs[i].pop(j)
                else:
                    j += 1

    def evaluate(self):
        if self.tar_file:
            tar_sketch = UnlabeledObject(np.concatenate([obj.get_strokes() for obj in self.tar_objs]))
        
        c , n = 0, 0
        tmp_sketch = []
        for sketch in self.test_objs:
            tmp = []
            for obj in sketch:
                tmp.append(obj.get_strokes())
                if len(obj.get_strokes()) == 1:
                    c += 1
                n += 1
            try:
                tmp_sketch.append(UnlabeledObject(np.concatenate((tmp))))
            except Exception as e:
                print(len(tmp), len(sketch))

        print("[ParsingEvaluation] Total number of objects: ", n)
        print("[ParsingEvaluation] Single stroke objects: ", c)
        print("[ParsingEvaluation] Single stroke/number of objects ration: ", c/n)

        test_sketch_lst = [UnlabeledObject(np.concatenate([obj.get_strokes() for obj in sketch])) for sketch in self.test_objs]

        st = time.time()
        clustering = DBScanClustering(test_sketch_lst)
        pred_test_objs, pred_test_segs = clustering.evaluate()
        # clustering = DensityClustering(tar_sketch, test_sketch_lst)
        # pred_tar_objs, pred_tar_segs, pred_test_objs, pred_test_segs = clustering.mut_execlusive_cluster()
        
        # tp = 0
        # for p_obj, lbl_lst in zip(pred_tar_objs, self.tar_lbs):
        #     # if the object belong to the target sketch
        #     if p_obj in self.tar_objs: 
        #             tp += 1
        #     # p_obj.visualize()
        
        # print("[ParsingEvaluation] info: target object segmentation accuracy", tp/len(self.tar_objs))
        
        n = sum([len(a) for a in pred_test_objs])
        print("[ParsingEvaluation] info: clustering predicted total", n, "objects")
        # find test set accuracy
        tp = tn = fp = fn = 0
        fn_objs, fp_objs = [], []
        print("target labels", self.tar_lbs)
        for t_objs, lbl_lst, p_objs in zip(self.test_objs, self.test_lbs, pred_test_objs):
            c = 0
            for t_obj, lbl in zip(t_objs, lbl_lst):
                # if the object belong to the target sketch
                if lbl in self.tar_lbs:
                    if t_obj in p_objs: 
                        tp += 1
                        c += 1
                    else:
                        fn += 1
                        fn_objs.append(t_obj)

                else:
                    if t_obj not in p_objs:
                        tn += 1
                    
            for obj in p_objs:
                # obj.visualize()
                if obj not in t_objs:
                    fp_objs.append(obj)
            
            
        fp = n - tp

        # obtain clustered 
        print("[ParsingEvaluation] info: Running time in hours: ", (time.time() - st) / 60 / 60)

        print("true positive:", tp)
        print("false positive:", fp)
        print("true negative:", tn)
        print("false negative:", fn)
        print("precision", (tp)/(tp+fp))
        print("recall", (tp)/(tp+fn))
        print("accuracy", (tp+tn)/(tp+fp+tn+fn))


        return tp, fp, tn, fn
