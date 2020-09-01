from RegistrationUtils import RegistrationUtils
from ObjectUtil import ObjectUtil
import numpy as np 
from progress.bar import Bar
from tqdm import tqdm
import os
import pathlib
from multiprocessing import Pool
import multiprocessing
import time
import pandas as pd

class Evaluation:
    inf = 1e9
    acceptable_labels = ['Triangle', 'Circle', 'Star', 'Diamond', 'Square', 'Star Bullet', 'Parallelogram Left', 'Parallelogram Right']
    target_labels = {'Circle':'Circle', 'Triangle':'Triangle', 'Star':'Star', 'Square':'Square', 'Diamond':'Square', 'Parallelogram Left':'Square', 'Parallelogram Right':'Square'}
    def __init__(self, prototypes, labels, re_sampling = 1.0):
        self.prototypes = prototypes
        self.labels = labels
        self.core_cnt = multiprocessing.cpu_count()
        self.re_sampling = re_sampling

    def add_file(self, file):
        objs, lbs = ObjectUtil.xml_to_UnlabeledObjects(file, re_sampling=self.re_sampling)
        for obj, label in zip(objs, lbs):
            if label in self.labels:
                ind = self.labels.index(label)
                self.prototypes[ind].append(obj)
            else:
                self.labels.append(label)
                self.prototypes.append([obj])

    def explore(self, directory, scale, pro_queue, labels_ind):
        # prepare queue of regiteration objects for multiprocessing
        for path in pathlib.Path(directory).iterdir():
            if path.is_dir():
                self.explore(path, scale, pro_queue, labels_ind)
            elif path.is_file():
                file = os.path.basename(path)
                if file.endswith(".xml"): 
                    try:
                        objs, lbs = ObjectUtil.xml_to_UnlabeledObjects(str(path), re_sampling=self.re_sampling)
                        for obj, label in zip(objs, lbs):
                            if label in self.acceptable_labels:
                                pro_queue.append(obj)
                                t_label = self.target_labels[label]
                                ind = self.labels.index(t_label)
                                labels_ind.append(ind)
                    except Exception as e: 
                        print("could not read file succefully " + file)
                        print(str(e))


    def start(self, path, scale):
        st = time.time()
        self.scale = scale
        pro_queue, labels_ind = [], []
        self.explore(path, scale, pro_queue, labels_ind)

        print("The number of objects to be evaluated are", len(labels_ind))
        self.labels.append('Unknown')
        conf_matrix=pd.DataFrame(np.zeros((len(self.acceptable_labels), len(self.labels))), columns=self.labels, index=self.acceptable_labels)
        # register all the objects using pooling
        res = []
        pool = Pool(self.core_cnt)
        for r in tqdm(pool.imap(self.evaluate_obj, pro_queue), total=len(labels_ind)):
            res.append(r)
        # with Pool(self.core_cnt) as p:
        #     res = list(p.map(self.evaluate_obj, pro_queue))

        for prediction, ind in zip(res, labels_ind):
            conf_matrix.iloc[ind,prediction] += 1
        
        pl = 0
        for i in range(len(self.labels) - 1):
            pl += conf_matrix.iloc[i,i]
        nl = len(labels_ind)

        print("Running time: ", time.time()-st)
        return pl / nl, conf_matrix

    def evaluate_obj(self, obj):
        tmp = []
        for ps in self.prototypes:
            tmp.append(self.inf)
            for o in ps:
                d = RegistrationUtils.identify_similarity(obj, o)
                tmp[-1] = min(tmp[-1], d)
        mn_ind = np.argmin(tmp)
        if tmp[mn_ind] > self.scale:
            return len(tmp)
        else:
            return mn_ind

