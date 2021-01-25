
import sys
sys.path.insert(0, '../')

from register.Registration import Registration
from utils.ObjectUtil import ObjectUtil
import os.path as path
import numpy as np

# find the pair-wise embeddings distance of two sketches
s = 6 # index of which sample to run the test on

org_path = 'input_directory/samples/test_samples/a' + str(s) + '.xml'
org_path = path.join(path.abspath(path.join(__file__ ,"../..")), org_path)
tar_path = 'input_directory/samples/test_samples/b' + str(s) + '.xml'
tar_path = path.join(path.abspath(path.join(__file__ ,"../..")), tar_path)

reg = Registration(org_path, tar_path, mn_stroke_len=3, re_sampling=1, flip=True, shift_target_y = 0)
embds = ObjectUtil.get_embedding(np.concatenate([reg.original_obj, reg.target_obj]))
org_embd = embds[:len(reg.original_obj)]
tar_embd = embds[len(reg.original_obj):]

for i, embd1 in enumerate(org_embd):
    for j, embd2 in enumerate(tar_embd):
        print("Embeddings distance between {}-{}:{}".format(reg.origninal_labels[i], reg.target_labels[j], np.linalg.norm(embd1 - embd2)))

t_labels = np.concatenate((reg.origninal_labels, reg.target_labels))
p_labels = ObjectUtil.classify(np.concatenate((reg.original_obj, reg.target_obj)))
print("Object True labels: ", t_labels)
print("Object predicted labels: ", p_labels)
tp = sum([a.lower() == b.lower() for a, b in zip(t_labels, p_labels)])
print("Accuracy score: {0}", tp/len(t_labels))
