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
        exp_id = 'testing_with_6params_pred_midresnet_basic_block'
    
    model_config = Config.default_model_config(exp_id)
    model_config.learning_rate = 0.005
    model_config.deacy_rate = 0.001
    model_config.n_files = 100
    model_config.k_select = 10
    model_config.epochs = 300
    model_config.re_sampling = 110
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

    model_config.dataset_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_B/Triangles')
    
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

    print("Number of repeated object:", (tot - len(objs)) / 2)
    

    random.seed(model_config.seed)
    val_inds = np.random.choice(np.arange(len(objs)), int(0.1 * len(objs)), replace=False)
    trn_inds = np.array([x for x in range(len(objs)) if x not in val_inds])
    print(len(objs))
    print(trn_inds)
    train_org_sketches, val_org_sketches, train_tar_sketches, val_tar_sketches  = [], [], [], []
    tmp_objs, tmp_labels = objs[trn_inds], labels[trn_inds]
    for obj, lbl in zip(tmp_objs, tmp_labels):
        if model_config.select_only_matched:
            matched_objs = tmp_objs[tmp_labels == lbl] # TODO test with non-matched objects
        else:
            matched_objs = tmp_objs[trn_inds]

        # choose k random matched objects
        matched_objs = random.choices(matched_objs, k=model_config.k_select)

        for obj2 in matched_objs:
            train_org_sketches.append(obj)
            train_tar_sketches.append(obj2)


    tmp_objs, tmp_labels = objs[val_inds], labels[val_inds]
    for obj, lbl in zip(tmp_objs, tmp_labels):
        if model_config.select_only_matched:
            matched_objs = tmp_objs[tmp_labels == lbl] # TODO test with non-matched objects
        else:
            matched_objs = tmp_objs[trn_inds]

        # choose k random matched objects
        matched_objs = random.choices(matched_objs, k=model_config.k_select)

        for obj2 in matched_objs:
            val_org_sketches.append(obj)
            val_tar_sketches.append(obj2)


    # # split train test
    # train_org_sketches, val_org_sketches, train_tar_sketches, val_tar_sketches = train_test_split(org_objs, tar_objs, random_state=model_config.seed, test_size=0.2)


    # save experiment configurations
    config_json = json.dumps(dict(model_config), indent=4)
    with open(model_config.config_path, 'w') as f:
        f.write(config_json)

    # redirect output to log
    if model_config.redirect_out:
        sys.stdout = open(model_config.log_path, 'w+')

    model = NNModel(model_config)

    if model_config.epochs > 0:
        model.fit(train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches)

    # # visualize model and save results
    model_visualizer.visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config)
    