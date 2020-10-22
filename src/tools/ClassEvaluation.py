from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
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

class ClassEvaluation:
    inf = 1e9
    available_labels = ['Triangle', 'Circle', 'Star', 'Diamond', 'Square', 'Star Bullet', 'Parallelogram Left', 'Parallelogram Right', 
    'Equals', 'Arrow Right', 'Arrow Up', 'Two Boxes', 'Two Boxes Null', 'Trapezoid Down', 'Trapezoid Up', 'Resistor Horizontal', 
    'Resistor Horizontal', 'Resistor Vertical', 'Battery Right', 'Battery Down', 'Plus', 'Minus', 'Cross', 'Arrow Right']
    # acceptable_labels = ['Triangle', 'Circle', 'Star', 'Diamond', 'Square', 'Parallelogram Left', 'Parallelogram Right']
    acceptable_labels = available_labels
    target_labels = {'Circle':'Circle', 'Triangle':'Triangle', 'Star':'Star', 'Square':'Square', 'Diamond':'Square', 
    'Star Bullet':'Star Bullet', 'Parallelogram Left':'Square', 'Parallelogram Right':'Square', 'Equals':'UNK', 
    'Arrow Right':'Arrow Right', 'Arrow Up':'Arrow Right', 'Two Boxes':'Two Boxes', 'Two Boxes Null':'Two Boxes Null',
    'Trapezoid Down':'Trapezoid Down', 'Trapezoid Up':'Trapezoid Down', 'Resistor Horizontal':'UNK', 
    'Resistor Vertical':'UNK', 'Battery Right':'UNK', 'Battery Down':'UNK', 'Plus':'UNK', 'Minus':'UNK',
    'Cross':'UNK'}

    def __init__(self, prototypes, labels, re_sampling = 1.0):
        self.prototypes = prototypes
        self.labels = labels
        self.core_cnt = multiprocessing.cpu_count()
        self.re_sampling = re_sampling
        self.labels_cnt = {}

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
                if str(path).endswith(".xml"): 
                    try:
                        objs, lbs = ObjectUtil.xml_to_UnlabeledObjects(str(path), re_sampling=self.re_sampling)
                        for obj, label in zip(objs, lbs):
                            if label in self.acceptable_labels:
                                if label not in self.labels_cnt:
                                    self.labels_cnt[label] = 1
                                else:
                                    self.labels_cnt[label] += 1
                                pro_queue.append(obj)
                                labels_ind.append(self.acceptable_labels.index(label))
                    except Exception as e: 
                        print("could not read file succefully " + path)
                        print(str(e))


    def start(self, path, scale):
        st = time.time()
        self.scale = scale
        pro_queue, labels_ind = [], []
        self.explore(path, scale, pro_queue, labels_ind)

        print("The number of objects to be evaluated are", len(labels_ind))
        for label in self.labels_cnt:
            print(label, self.labels_cnt[label])
        self.labels.append('UNK')
        k_cnt, k_start, k_step = 31, 20, 1
        conf_matrix= [pd.DataFrame(np.zeros((len(self.acceptable_labels), len(self.labels))), columns=self.labels, index=self.acceptable_labels) for _ in range(k_cnt)]
        # register all the objects using pooling
        res = []
        pool = Pool(self.core_cnt)
        for r in tqdm(pool.imap(self.evaluate_obj, pro_queue), total=len(labels_ind)):
            res.append(r)
        # with Pool(self.core_cnt) as p:
        #     res = list(p.map(self.evaluate_obj, pro_queue))
        
        pl = np.zeros((k_cnt))
        for p, ind in zip(res, labels_ind):
            p_ind, p_val = p[0], p[1]
            for i in range(k_cnt):
                if p_val >= k_start + i * k_step:
                    prd = self.labels.index('UNK')
                else:
                    prd = p_ind
                conf_matrix[i].iloc[ind,prd] += 1
                label = self.acceptable_labels[ind]
                t_label = self.target_labels[label]
                t_ind = self.labels.index(t_label)
                if t_ind == prd:
                    pl[i] += 1
        nl = len(labels_ind)
        print("Running time in hours: ", (time.time() - st) / 60 / 60)

        for i in range(k_cnt):
            print("Test with scale ", k_start + k_step * i)
            print("Prediction Accuracy is: ", pl[i] / nl)
            print("Confusion matrix:")
            print(conf_matrix[i])

    def evaluate_obj(self, obj):
        tmp = []
        for ps in self.prototypes:
            tmp.append(self.inf)
            for o in ps:
                d = RegistrationUtils.identify_similarity(obj, o)
                tmp[-1] = min(tmp[-1], d)
        mn_ind = np.argmin(tmp)
        return [mn_ind, tmp[mn_ind]]
