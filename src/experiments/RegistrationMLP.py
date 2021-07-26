import sys
sys.path.insert(0, '../')

import numpy as np
from utils.Config import Config 
from registrationNN.models import NNModel, model_visualizer
from utils.ObjectUtil import ObjectUtil
from sklearn.model_selection import train_test_split
import os
from munch import Munch
import time
import random
import json 
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exp_id = str(sys.argv[1])
    else:
        exp_id = 'reflectionQuestion-Transformer-with-penalty'
    
    model_config = Config.default_model_config(exp_id)
    model_config.learning_rate = 1e-4
    model_config.n_files = 5000
    model_config.k_select = 100
    model_config.epochs = 500
    model_config.comment = 'no penalty on the movements'

    print(f"[RegisterationMLP.py] {time.ctime()}: Expermint {model_config.exp_id} started")

    org_objs, tar_objs = [], []
    objs, labels = ObjectUtil.extract_objects_from_directory(model_config.dataset_path,
                                                            n_files=model_config.n_files,
                                                            acceptable_labels=model_config.obj_accepted_labels)
    labels, objs = np.asarray(labels), np.asarray(objs)

    for obj, lbl in zip(objs, labels):
        matched_objs = objs[labels == lbl] # TODO test with non-matched objects

        # choose k random matched objects
        matched_objs = random.choices(matched_objs, k=model_config.k_select)

        for obj2 in matched_objs:
            org_objs.append(obj)
            tar_objs.append(obj2)


    # split train test
    train_org_sketches, val_org_sketches, train_tar_sketches, val_tar_sketches = train_test_split(org_objs, tar_objs, random_state=model_config.seed, test_size=0.2)


    # save experiment configurations
    config_json = json.dumps(dict(model_config), indent=4)
    with open(model_config.config_path, 'w') as f:
        f.write(config_json)

    # redirect output to log
    sys.stdout = open(model_config.log_path, 'w+')

    model = NNModel(model_config)
    model.fit(train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches)

    # # visualize model and save results
    model_visualizer.visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config)
    