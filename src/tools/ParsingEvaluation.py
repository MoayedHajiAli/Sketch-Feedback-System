from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
from tools.StrokeClustering import DensityClustering
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

    def __init__(self, test_dir, tar_file, n_files=-1, re_sampling=0.0):
        self.test_dir = test_dir
        self.tar_file = tar_file

        # obtain all objects with their indicies
        self.tar_objs, self.tar_lbs, self.tar_segs = ObjectUtil.xml_to_IndexedUnlabeledObjects(tar_file, re_sampling=re_sampling)
        print("[ParsingEvaluation] taget object contains", len(self.tar_objs), "objects")
        self.test_objs, self.test_lbs, self.test_segs = self.explore(test_dir, n_files=n_files)
        print("[ParsingEvaluation] testing directory contains", sum([len(a) for a in self.test_objs]), "objects")
        self.n_files = n_files


    def explore(self, directory, n_files=-1, re_sampling=0.0):
        # prepare queue of regiteration objects for multiprocessing
        unorderd_objs = 0
        objs, labels, segs = [], [], []

        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                self.explore(path)
            elif path.is_file():
                if n_files != -1 and len(objs) > n_files:
                    break
                if str(path).endswith(".xml"): 
                    try:
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
                        objs.append(tmp_obj)
                        labels.append(tmp_lbl)
                        segs.append(set(tmp_segs))
                    except Exception as e: 
                        print("ParsingEvaluation] error: could not read file succefully " + str(path))
                        print(str(e))

        print("[ParsingEvaluation] warn:", unorderd_objs, "unordered objects were found and discarded" )
        return objs, labels, segs

    def evaluate(self):
        st = time.time()
        
        clustering = DensityClustering.fromDir(self.tar_file, self.test_dir, n=self.n_files)
        pred_tar_segs, pred_test_segs = clustering.mut_execlusive_cluster()

        n = sum([len(a) for a in pred_test_segs])
        print("[ParsingEvaluation] info: clustering predected total", n, "objects")
        # find test set accuracy
        tp = tn = fp = fn = 0
        for t_segs, lbl_lst, p_segs in zip(self.test_segs, self.test_lbs, pred_test_segs):
            # print("true segments", [seg for seg, lbl in zip(t_segs, lbl_lst) if lbl in self.tar_lbs])
            # print("predicted segments", p_segs)
            # print("")
            c = 0
            for t_seg, lbl in zip(t_segs, lbl_lst):
                # if the object belong to the target sketch
                if lbl in self.tar_lbs:
                    if t_seg in p_segs:
                        tp += 1
                        c += 1
                    else:
                        fn += 1
                else:
                    if t_seg not in p_segs:
                        tn += 1
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
