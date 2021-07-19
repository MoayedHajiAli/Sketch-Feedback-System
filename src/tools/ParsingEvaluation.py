from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
from tools.StrokeClustering import DensityClustering, DBSCAN_segmentation
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
    """
    Pefroming an experiment on object-level segmentation
    """
    def __init__(self, config):
        self.config = config

        # obtain all objects with their labels and index of start and end stroke (all unordered objects will be discarded)
        if self.config.tar_file:
            self.tar_objs, self.tar_lbs, self.tar_segs = ObjectUtil.xml_to_IndexedUnlabeledObjects(self.config.tar_file, re_sampling=self.config.re_sampling)
            print("[ParsingEvaluation] target object contains", len(self.tar_objs), "objects")
        
        self.test_objs, self.test_lbs, self.test_segs = self.extract_objs_from_dir(self.config.test_dir, n_files=self.config.n_files, re_sampling=self.config.re_sampling)
        self.test_objs, self.test_lbs, self.test_segs = \
                        np.asarray(self.test_objs), np.asarray(self.test_lbs), np.asarray(self.test_segs)
        print("[ParsingEvaluation] testing directory contains", sum([len(a) for a in self.test_objs]), "objects")

        # remove unwanted objects such as number
        self.clean(self.config.accepted_labels, verbose=self.config.verbose)

        if self.config.verbose >= 1:
            if self.config.tar_file:
                print("[ParsingEvaluation] after cleaning, taget object contains", len(self.tar_objs), "objects")
            
            print("[ParsingEvaluation] after cleaning, testing directory contains", sum([len(a) for a in self.test_objs]), "objects")

    def extract_objs_from_dir(self, directory, n_files=-1, re_sampling=0.0):
        # explore n_files in the directory. extracts the objects and their stroke's ordering 
        unorderd_objs = 0

        # objs : (N x M) M ordered objects for N sketches
        # labels: (N x M) for each object m in sketch n, store its label
        objs, labels, segs = [], [], []

        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                self.extract_objs_from_dir(path)
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

    def clean(self, accepted_labels, verbose=0):
        """Clean the target and orignal storkes by removing unrelated objects such as handwritting
        """
        if self.config.tar_file:
            i = 0
            while i < len(self.tar_objs):
                if self.tar_lbs[i] not in accepted_labels:
                    self.tar_lbs.pop(i)
                    self.tar_objs.pop(i)
                    self.tar_segs.pop(i)
                else:
                    i += 1
        
        inds = []
        for i in range(len(self.test_objs)):
            j = 0
            while j < len(self.test_objs[i]):
                if self.test_lbs[i][j] not in accepted_labels:
                    self.test_lbs[i].pop(j)
                    self.test_objs[i].pop(j)
                    self.test_segs[i].pop(j)
                else:
                    j += 1
            if j == 0 and verbose >= 2:
                print("[ParsingEvaluation] warn: found a sketch with no object in the accepted labels")
            if j > 0:
                inds.append(i)
        self.test_objs = np.asarray(self.test_objs)[inds]
        self.test_segs = np.asarray(self.test_segs)[inds]
        self.test_lbs = np.asarray(self.test_lbs)[inds]

        if verbose >= 1:
                print("[ParsingEvaluation] info: finished cleaning sketches")
    


    def evaluate(self):
        # comine the objects to their sketches (this way unlabeled objects will be removed from the sketches)
        if self.config.tar_file:
            tar_sketch = UnlabeledObject(np.concatenate([obj.get_strokes() for obj in self.tar_objs]))
        
        c, n = 0, 0
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
                print("[ParsingEvaluation] error:", len(tmp), len(sketch))

        if self.config.verbose >= 1:
            print("[ParsingEvaluation] Total number of objects: ", n)
            print("[ParsingEvaluation] Single stroke objects: ", c)
            print("[ParsingEvaluation] Single stroke/number of objects ration: ", c/n)

        # combine the objects to sketches
        test_sketch_lst = [UnlabeledObject(np.concatenate([obj.get_strokes() for obj in sketch])) for sketch in self.test_objs]

        st = time.time()
        
        clustering = DBSCAN_segmentation(test_sketch_lst, self.config)
        pred_test_objs, pred_test_segs = clustering.segment(np.concatenate(self.test_objs))
        
        n = sum([len(a) for a in pred_test_objs])
        if self.config.verbose >= 1:
            print("[ParsingEvaluation] info: clustering predicted total", n, "objects")
        
        
        print(f"[ParsingEvaluation] info: Running time:{(time.time() - st) / 60 } Minutes")

