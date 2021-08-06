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
        exp_id = 'after-decomposition-penalty'
    
    model_config = Config.default_model_config(exp_id)
    model_config.learning_rate = 1e-3
    model_config.n_files = 15
    model_config.k_select = 5
    model_config.epochs = 200
    model_config.re_sampling = 0.5
    model_config.comment = 'no penalty on the movements'
    model_config.select_only_matched = True
    model_config.obj_accepted_labels = ['Triangle']
    model_config.redirect_out = False

    model_config.load = False
    model_config.load_ckpt = False
    model_config.save = True
    model_config.save_ckpt = True
    model_config.vis_transformation = False
    model_config.num_vis_samples = 5
    
    print(f"[RegisterationMLP.py] {time.ctime()}: Expermint {model_config.exp_id} started")

    org_objs, tar_objs = [], []
    objs, labels = ObjectUtil.extract_objects_from_directory(model_config.dataset_path,
                                                            n_files=model_config.n_files,
                                                            acceptable_labels=model_config.obj_accepted_labels,
                                                            re_sampling = model_config.re_sampling)
    labels, objs = np.asarray(labels), np.asarray(objs)

    # validate that objects are distincts 
    tot = 0
    for obj1 in objs:
        for obj2 in objs:
            if obj1 == obj2:
                tot += 1

    print(len(objs), tot)
    
    random.seed(model_config.seed)
    for obj, lbl in zip(objs, labels):
        if model_config.select_only_matched:
            matched_objs = objs[labels == lbl] # TODO test with non-matched objects
        else:
            matched_objs = objs

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
    if model_config.redirect_out:
        sys.stdout = open(model_config.log_path, 'w+')

    model = NNModel(model_config)
    model.fit(train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches)

    # # visualize model and save results
    model_visualizer.visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config)
    