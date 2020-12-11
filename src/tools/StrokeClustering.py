from sketch_object.UnlabeledObject import UnlabeledObject
from utils.ObjectUtil import ObjectUtil
import numpy as np
import copy
import pathlib
import os
import sklearn.decomposition.pca
import matplotlib.pyplot as plt

class DensityClustering:
    """Perfore density based clustering of all possible stroke cominations of the objects around all combinations of strokes 
    of the target object. This is follwed by a process of selection of the clusters with the most number of point, and then elimination
    of points that have intersected strokes with the selected points. Sketchformer embeddings are used for moving the sketches into
    the embedding space.

    Assumptions:
    1- combinations are taken only from contigues subset of strokes
    2- object length is limited to 200 (the input length of sketchformer)
    """

    def __init__(self, tar_obj:UnlabeledObject, objs):
        # obtain all possible combinations of the target object
        self.tar_objs = self._get_combinations(tar_obj, -1)

        # obtain the number of sketches to be clustered
        self.N = len(objs)

        self.org_objs = []
        for i in range(len(objs)):
            tmp = self._get_combinations(objs[i], i)
            for obj in tmp:
                self.org_objs.append(obj)
        
        self.org_objs = np.array(self.org_objs)
    
    @classmethod
    def fromDir(cls, tar_file, dir, n=None):
        """Explore the directory and cluster all the sketches inside the directory

        Args:
            tar_obj(UnlabeledObject)
            dir (str): [description]
        """

        tar_obj = UnlabeledObject(ObjectUtil.xml_to_strokes(tar_file))

        objs = []
        for path in pathlib.Path(dir).iterdir():
            if path.is_file():
                file = os.path.basename(path)
                if file.endswith(".xml"): 
                    try:
                        objs.append(UnlabeledObject(ObjectUtil.xml_to_strokes(str(path))))
                        for st in objs[-1].get_strokes():
                            assert(len(st) >= 5)
                        if n and len(objs) >= n:
                            break
                    except:
                        print("could not convert file " + file)
        
        return cls(tar_obj, objs)



    def mut_inclusive_cluster(self, eps = 18):
        # obtain embeddings for all objects
        print("Obtaining embeddings")
        self.tar_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.tar_objs])
        print("target embeddings obtained")
        self.org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        print("original embeddings obtained")
        self.org_seg, self.tar_seg = [[] for _ in range(self.N)], []

        # cluster
        clusters = [[] for _ in range(len(self.tar_objs))]
        for i in range(len(clusters)):
            for j in range(len(self.org_objs)):
                if self._embd_dist(self.tar_embds[i], self.org_embds[j]) <= eps:
                    clusters[i].append(self.org_objs[j])
        
        # select and eleminate
        while True:
            # find the cluster with the largest number of points
            ind, mx = -1, -1
            for i in range(len(clusters)):
                self.tar_objs[i]['obj'].visualize()
                print(len(clusters[i]))
                if len(clusters[i]) > mx:
                    mx, ind = len(clusters[i]), i
            
            if mx <= 0:
                break
            
            l, r = self.tar_objs[ind]['l'], self.tar_objs[ind]['r']
            self.tar_seg.append((l, r))
            print("l, r", l, r)
            selected_cluster = clusters[ind]

            print("A new cluster found with max", mx)
            # self.tar_objs[ind]['obj'].visualize()

            # filter all intersected clusters
            clusters = [clusters[i] if not self._check_intersection(l, r, self.tar_objs[i]['l'], self.tar_objs[i]['r']) else [] for i in range(len(clusters))]


            while selected_cluster:
                selected_tpl = selected_cluster[0]
                p, l, r = selected_tpl['id'], selected_tpl['l'], selected_tpl['r']
                # selected_tpl['obj'].visualize()
                self.org_seg[p].append((l, r))

                # filter all intersected sketches
                for i in range(len(clusters)):
                    clusters[i] = [obj for obj in clusters[i] if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
                
                # filter intersected sketches in the same cluster
                selected_cluster = [obj for obj in selected_cluster if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
            
        return self.tar_seg, self.org_seg
    

    def mut_execlusive_cluster(self, eps=18, per=0.5):
        
        self.tar_objs = sorted(self.tar_objs, key=lambda a: (a['l'] - a['r']))

        # obtain embeddings for all objects
        print("Obtaining embeddings")
        self.tar_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.tar_objs])
        print("target embeddings obtained")
        self.org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        print("original embeddings obtained")
        self.org_seg, self.tar_seg = [[] for _ in range(self.N)], []

        # cluster
        clusters = [[] for _ in range(len(self.tar_objs))]
        for j in range(len(self.org_objs)):
            mn, ind = 1e9, -1
            for i in range(len(clusters)):
                d = self._embd_dist(self.tar_embds[i], self.org_embds[j])
                if d <= mn:
                    mn, ind = d, i
            
            if mn <= eps:
                self.org_objs[j]['dist'] = mn
                clusters[ind].append(self.org_objs[j])

                
        
        # select and eleminate
        while True:
            # find the cluster with the largest number of points
            ind, mx = -1, -1
            for i in range(len(clusters)):
                if len(clusters[i]) > self.N * per:
                    mx, ind = len(clusters[i]), i
                    break
            
            if mx <= 0:
                break
            
            l, r = self.tar_objs[ind]['l'], self.tar_objs[ind]['r']
            self.tar_seg.append((l, r))
            selected_cluster = clusters[ind]

            print("A new cluster found with max", mx)
            print("l, r", l, r)
            # self.tar_objs[ind]['obj'].visualize()

            # filter all intersected clusters
            clusters = [clusters[i] if not self._check_intersection(l, r, self.tar_objs[i]['l'], self.tar_objs[i]['r']) else [] for i in range(len(clusters))]

            # sort the cluster according to the embedding distance
            selected_cluster = sorted(selected_cluster, key=lambda a : a['dist'])
            while selected_cluster:
                selected_tpl = selected_cluster[0]
                p, l, r = selected_tpl['id'], selected_tpl['l'], selected_tpl['r']
                # selected_tpl['obj'].visualize()
                self.org_seg[p].append((l, r))

                # filter all intersected sketches
                for i in range(len(clusters)):
                    clusters[i] = [obj for obj in clusters[i] if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
                
                # filter intersected sketches in the same cluster
                selected_cluster = [obj for obj in selected_cluster if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
            
        return self.tar_seg, self.org_seg


    def _visualize_clusters(clusters):
        pca = PCA(n_components = 2)
        pca_res = pca.fit_transform(clusters)
        x, y = pca_res[:, 0], pca_res[:,1]
        plt.figure(figsize=(16,10))
        plt.plot(x, y)
        plt.show()


    def _check_intersection(self, l1, r1, l2, r2):
        return not ((r1 < l2) or (l1 > r2))


    def _embd_dist(self, embd1, embd2):
        return np.linalg.norm((embd2 - embd1))
    
    def _objs_dist(self, obj1, obj2):
        l1, r1 = min(obj1.get_x()), max(obj1.get_x())
        l2, r2 = min(obj2.get_x()), max(obj2.get_x())

        if l2 > l1:
            d1 =  l2 - r1
        else: 
            d1 =  l1 - r2
        
        l1, r1 = min(obj1.get_y()), max(obj1.get_y())
        l2, r2 = min(obj2.get_y()), max(obj2.get_y())

        if l2 > l1:
            d2 =  l2 - r1
        else: 
            d2 =  l1 - r2
        
        return max(d1, d2)


    def _get_combinations(self, obj:UnlabeledObject, id, mx_dis = 10):
        """for a given object, obtain all possible combinations of strokes

        Args:
            obj ([UnlabeledObject]): [description]

        Returns: list of dict, each of have information of obj, id, l, r
        """
        stroke_lst = obj.get_strokes()
        res = []
        for i in range(len(stroke_lst)):
            tmp, total_len, last_obj = [], 0, None
            for j in range(i, len(stroke_lst)):
                if total_len > 200 or (j != i and self._objs_dist(UnlabeledObject([stroke_lst[j-1]]), UnlabeledObject([stroke_lst[j]])) > mx_dis):
                    break
                tmp.append(stroke_lst[j].get_copy())
                total_len += len(stroke_lst[j])
                last_obj = UnlabeledObject(copy.deepcopy(tmp))
                res.append({"obj": last_obj, "id":id, "l":i, "r":j})
        
        return np.array(res)
