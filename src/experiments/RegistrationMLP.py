import sys
sys.path.insert(0, '../')

import numpy as np
from utils.Config import Config 
from registrationNN.models import registration_model, model_visualizer
from utils.ObjectUtil import ObjectUtil
from sklearn.model_selection import train_test_split
import os
from munch import Munch
import time
import random

# # train a MLP for registration
# dir = 'ASIST_Dataset/Data/Data_B/Circle'
# dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
# objs, labels, val_objs, val_labels = [], [], [], []
# n, N, val_N = 0, 400, 0
# for file_path in pathlib.Path(dir).iterdir():
#     if n >= val_N:
#         break
#     a, b = ObjectUtil.xml_to_UnlabeledObjects(str(file_path))
#     if n > N:
#         val_objs.extend(a)
#         val_labels.extend(b)
#     elif n > 400:
#         objs.extend(a)
#         labels.extend(b)
#     n += len(a)

# # train a MLP for registration
# dir = 'ASIST_Dataset/Data/Data_B/Triangles'
# dir = path.join(path.abspath(path.join(__file__ ,"../../..")), dir)
# n, N, val_N = 0, 1000, 1050
# for file_path in pathlib.Path(dir).iterdir():
#     if n >= val_N:
#         break
#     a, b = ObjectUtil.xml_to_UnlabeledObjects(str(file_path))
#     if n > N:
#         val_objs.extend(a)
#         val_labels.extend(b)
#     elif n > 1000:
#         objs.extend(a)
#         labels.extend(b)
#     n += len(a)
# print("Total size: {0}".format(n))
# print(len(objs))
# fig, axs = plt.subplots(4, 4)
# for i, obj in enumerate(objs):
#     obj.visualize(ax=axs[int(i/4), int(i%4)], show=False)



model_config = Config.default_model_config(10)
model_config.n_files = 200
model_config.k_select = 10
model_config.epochs = 200
model_config.num_vis_samples = 5 
model_config.obj_accepted_labels = ['Circle', 'Star', 'Triangle']

print(f"[RegisterationMLP.py] {time.ctime()}: Expermint {model_config.exp_id} started")

org_objs, tar_objs = [], []
objs, labels = ObjectUtil.extract_objects_from_directory(model_config.dataset_path, n_files=model_config.n_files, \
                acceptable_labels=model_config.obj_accepted_labels)
labels, objs = np.asarray(labels), np.asarray(objs)

for obj, lbl in zip(objs, labels):
    matched_objs = objs[labels == lbl]

    # choose k random matched objects
    matched_objs = random.choices(matched_objs, k=model_config.k_select)

    for obj2 in matched_objs:
        org_objs.append(obj)
        tar_objs.append(obj2)


# split train test
train_org_sketches, val_org_sketches, train_tar_sketches, val_tar_sketches = train_test_split(org_objs, tar_objs, random_state=model_config.seed, test_size=0.2)

# redirect output to log
# sys.stdout = open(os.path.join(model_config.exp_dir, 'log.out'), 'w+')

model = registration_model(model_config)
model.fit(train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches)

# visualize model and save results
model_visualizer.visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config)
