


 # find the pair-wise embeddings distance of two sketches
reg = Registration('./input_directory/samples/test_samples/a' + str(s) + '.xml', './input_directory/samples/test_samples/b' + str(s) + '.xml', mn_stroke_len=3, re_sampling=1, flip=True, shift_target_y = 0)
embds = ObjectUtil.get_embedding(np.concatenate([reg.original_obj, reg.target_obj]))
org_embd = embds[:len(reg.original_obj)]
tar_embd = embds[len(reg.original_obj):]

for i, embd1 in enumerate(org_embd):
    for j, embd2 in enumerate(tar_embd):
    print("Embeddings distance between {}-{}:{}".format(reg.origninal_labels[i], reg.target_labels[j], np.linalg.norm(embd1 - embd2)))


print("Object predicted labels: ",  ObjectUtil.classify(np.concatenate((reg.original_obj, reg.target_obj))))
