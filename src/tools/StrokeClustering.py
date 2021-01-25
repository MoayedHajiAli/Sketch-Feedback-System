from sketch_object.UnlabeledObject import UnlabeledObject
from utils.ObjectUtil import ObjectUtil
from utils.RegistrationUtils import RegisterTwoObjects, RegistrationUtils
from animator.SketchAnimation import SketchAnimation
import numpy as np
import copy
import pathlib
import os
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import random as rnd


class DensityClustering:
    """Perform density based clustering of all possible stroke cominations of the objects around all combinations of strokes 
    of the target object. This is follwed by a process of selection of the clusters with the most number of point, and then elimination
    of points that have intersected strokes with the selected points. Sketchformer embeddings are used for moving the sketches into
    the embedding space.

    Assumptions:
    1- combinations are taken only from contigues subset of strokes
    2- object length is limited to 200 (the input length of sketchformer)
    """

    def __init__(self, tar_obj:UnlabeledObject, objs):
        """set the inital combinations obtained from the target sketch and test sketches

        Args:
            tar_obj (UnlabeledObject): a target sketch where meaningful objects will be extracted from
            objs (list of UnlabledObject): list of sketches that we want to segment
        """
        # TODO: remove stroke visualization
        # tar_obj.visualize()

        # for obj in objs:
        #     obj.visualize()

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
        centroids will be taken only from the strokes combinations of the target sketch in the target-file

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
        print("[StrokeClustering] Obtaining embeddings")
        self.tar_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.tar_objs])
        print("[StrokeClustering] Target embeddings obtained")
        self.org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        print("[StrokeClustering] Original embeddings obtained")
        self.org_seg, self.tar_seg = [[] for _ in range(self.N)], []
        self.org_ret_objs, self.tar_ret_objs = [[] for _ in range(self.N)], []


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
                if len(clusters[i]) > mx:
                    mx, ind = len(clusters[i]), i
            
            if mx <= 0:
                break
            
            l, r = self.tar_objs[ind]['l'], self.tar_objs[ind]['r']
            self.tar_seg.append((l, r))
            self.tar_ret_objs.append(self.tar_objs[ind]['obj'])
            selected_cluster = clusters[ind]

            print("[StrokeClustering] A new cluster found with max", mx)
            self.tar_objs[ind]['obj'].visualize()

            # filter all intersected clusters
            clusters = [clusters[i] if not self._check_intersection(l, r, self.tar_objs[i]['l'], self.tar_objs[i]['r']) else [] for i in range(len(clusters))]


            while selected_cluster:
                selected_tpl = selected_cluster[0]
                # each sketch has a distinct id so that we eliminate the intersected combination only from the same
                # sketch.
                p, l, r = selected_tpl['id'], selected_tpl['l'], selected_tpl['r']
                # selected_tpl['obj'].visualize()
                self.org_seg[p].append((l, r))
                self.org_ret_objs[p].append(selected_tpl['obj'])

                # filter all intersected sketches in all clusters
                for i in range(len(clusters)):
                    clusters[i] = [obj for obj in clusters[i] if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
                
                # filter intersected sketches in the same cluster
                selected_cluster = [obj for obj in selected_cluster if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
            
        return self.tar_ret_objs, self.tar_seg, self.org_ret_objs, self.org_seg
    

    def mut_execlusive_cluster(self, eps=18, per=0.4):
        # store object decreasengly according to the number of strokes
        self.tar_objs = sorted(self.tar_objs, key=lambda a: (a['l'] - a['r']))

        # obtain embeddings for all objects
        print("Obtaining embeddings...")
        self.tar_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.tar_objs])
        print("target embeddings obtained")
        self.org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        print("original embeddings obtained")
        self.org_seg, self.tar_seg = [[] for _ in range(self.N)], []
        self.org_ret_objs, self.tar_ret_objs = [[] for _ in range(self.N)], []

        # initialize empty clusters for all target possible objects
        clusters = [[] for _ in range(len(self.tar_objs))]

        for j in range(len(self.org_objs)):
            # find the cluster with the minimum distance
            # print("Choosing new cluster for object")
            # self.org_objs[j]['obj'].visualize()
            mn, ind= 1e9, -1
            for i in range(len(clusters)):
                d = self._embd_dist(self.tar_embds[i], self.org_embds[j])
                # print("cluster embedding", d)
                # self.tar_objs[i]['obj'].visualize()
                if d <= mn and d < eps:
                    mn, ind = d, i
                    break
            
            if mn <= eps:
                self.org_objs[j]['dist'] = mn
                clusters[ind].append(self.org_objs[j])
                # print("[StrokeClustering] adding a new object to cluster {0}".format(ind))
                # # TODO: remove visualization
                # print(mn)
                # self.org_objs[j]['obj'].visualize()
                # self.tar_objs[ind]['obj'].visualize()

                
        # visualizing clusters
        # for i in range(len(self.tar_objs)):
        #     if len(clusters[i]) > 0:
        #         print("new cluster")
        #         self.tar_objs[i]['obj'].visualize()
        #         for tpl in clusters[i]:
        #             tpl['obj'].visualize()
                

        # select and eleminate
        x = 0
        while True:
            # find the cluster with the largest number of points
            ind, mx = -1, -1
            for i in range(len(clusters)):
                if len(clusters[i]) >= self.N * per:
                    mx, ind = len(clusters[i]), i
                    break
            
            # terminate if all clusters are empty
            if mx <= 0:
                break
            l, r = self.tar_objs[ind]['l'], self.tar_objs[ind]['r']
            self.tar_seg.append((l, r))
            self.tar_ret_objs.append(self.tar_objs[ind]['obj'])
            selected_cluster = clusters[ind]

            print("A new cluster found with max", mx)
            # self.tar_objs[ind]['obj'].visualize()
            # self.tar_objs[ind]['obj'].visualize()

            # filter all intersected clusters
            clusters = [clusters[i] if not self._check_intersection(l, r, self.tar_objs[i]['l'], self.tar_objs[i]['r']) else [] for i in range(len(clusters))]

            # sort the cluster according to the embedding distance
            selected_cluster = sorted(selected_cluster, key=lambda a : a['dist'])

            # sort the cluster (decreasingly) according to the number of strokes
            selected_cluster = sorted(selected_cluster, key=lambda a : a['l'] - a['r'])

            # x += 1
            # print(x)
            # if x == 3:
            #     for tpl in selected_cluster:
            #         tpl['obj'].visualize()


            while selected_cluster:
                selected_tpl = selected_cluster[0]
                # each sketch has a distinct id so that we eliminate the intersected combination only from the same
                # sketch.
                p, l, r = selected_tpl['id'], selected_tpl['l'], selected_tpl['r']
                
                # print(selected_tpl['dist'])
                # selected_tpl['obj'].visualize()
                
                self.org_seg[p].append((l, r))
                self.org_ret_objs[p].append(selected_tpl['obj'])

                # filter all intersected sketches from all clusters
                for i in range(len(clusters)):
                    clusters[i] = [obj for obj in clusters[i] if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
                
                # filter intersected sketches in the same cluster
                selected_cluster = [obj for obj in selected_cluster if (obj['id'] != p) or not self._check_intersection(l, r, obj['l'], obj['r'])]
            
        return self.tar_ret_objs, self.tar_seg, self.org_ret_objs, self.org_seg


    def reg_based_mut_execlusive_cluster(self, eps=50, per=0.5):
        self.tar_objs = sorted(self.tar_objs, key=lambda a: (a['l'] - a['r']))
        self.org_seg, self.tar_seg = [[] for _ in range(self.N)], []

        # cluster
        clusters = [[] for _ in range(len(self.tar_objs))]
        # TODO: remove the fixed start index (was used to testing only)
        for j in range(4, len(self.org_objs)):
            mn, ind = 1e9, -1
            for i in range(len(clusters)):
                d, p = RegisterTwoObjects(self.org_objs[j]['obj'], self.tar_objs[i]['obj'] , RegistrationUtils.total_transformation_cost).optimize()
                print(d)
                if d <= eps:
                    self.org_objs[j]['obj'].visualize()
                    self.tar_objs[i]['obj'].visualize()
                    animation = SketchAnimation([self.org_objs[j]['obj']], [self.tar_objs[i]['obj']])
                    # print(RegistrationUtils.calc_dissimilarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p), target_dis=False))
                    animation.seq_animate_all([p], save=False, file="./test_videos/example7-obj3-4.mp4")
                    plt.show()

                if d <= mn:
                    mn, ind = d, i
            
            if mn <= eps:
                self.org_objs[j]['obj'].visualize()
                self.tar_objs[ind]['obj'].visualize()
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


    def _get_combinations(self, obj:UnlabeledObject, id, mx_dis = 30):
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



class DBScanClustering:
    """object-level segmentation of a set of sketches
    """
    def __init__(self, objs):
        """set the inital combinations obtained from the target sketch and test sketches

        Args:
            objs (list of UnlabledObject): list of sketches that we want to segment
        """

        # obtain the number of sketches to be clustered
        self.N = len(objs)

        self.org_objs = []
        for i in range(len(objs)):
            tmp = self._get_combinations(objs[i], i)
            for obj in tmp:
                self.org_objs.append(obj)
        
        self.org_objs = np.array(self.org_objs)
    
    @classmethod
    def fromDir(cls, dir, n=None):
        """Explore the directory and cluster all the sketches inside the directory
        centroids will be taken only from the strokes combinations of the target sketch in the target-file

        Args:
            dir (str): a directoy where the sketches exits
        """
        objs = []
        for path in pathlib.Path(dir).iterdir():
            if path.is_file():
                file = os.path.basename(path)
                if file.endswith(".xml"): 
                    try:
                        objs.append(UnlabeledObject(ObjectUtil.xml_to_strokes(str(path))))
                        if n and len(objs) >= n:
                            break
                    except:
                        print("could not convert file " + file)
        
        return cls(objs)

    def _get_combinations(self, obj:UnlabeledObject, id, mx_dis = 30):
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

    def evaluate(self, mn_samples=0.1, eps=8):
        if mn_samples <= 1:
            mn_samples = int(self.N * mn_samples)
        
        # obtain embeddings
        org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        print("original embeddings obtained")

        # obtain tsne embeddings 
        tsne_embeddings = TSNE(2).fit_transform(org_embds)

        # plot embeddings
        plt.plot(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 'k.')
        plt.show()

        labels = np.asarray(DBSCAN(eps=eps, min_samples=mn_samples).fit(org_embds).labels_)
        n_clusters = max(labels) + 1
        clusters = []
        for i in range(n_clusters):
            clusters.append(np.where(labels==i)[0])
            # plotting the cluster with TSNE 
            plt.plot(tsne_embeddings[clusters[-1], 0], tsne_embeddings[clusters[-1], 1], '.')

        plt.show()    
        fig, axs = plt.subplots(n_clusters, 5)    
        for i, cluster in enumerate(clusters):
            axs[i, 0].set_title(str(len(cluster)))
            inds = rnd.choices(range(len(cluster)), k=5)
            for j, ind in enumerate(inds):
                self.org_objs[cluster[ind]]['obj'].visualize(ax=axs[i, j], show=False)
                axs[i, j].set_axis_off()
        plt.show()

        ret_objs = []
        print(clusters)

        
