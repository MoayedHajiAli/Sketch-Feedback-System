from sketch_object.UnlabeledObject import UnlabeledObject
from utils.ObjectUtil import ObjectUtil
from utils.RegistrationUtils import RegisterTwoObjects, RegistrationUtils
from animator.SketchAnimation import SketchAnimation
import numpy as np
import copy
import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
# from shapely.geometry import Point as shaplypt
# from shapely.ops import cascaded_union
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



class DBSCAN_segmentation:
    """object-level segmentation of a set of sketches
    """
    def __init__(self, objs, config):
        """set the inital combinations obtained from the target sketch and test sketches

        Args:
            objs (list of UnlabledObject): list of sketches that we want to segment
        """

        # sketches are assumed to be stored in a format of objects
        # obtain the number of sketches to be clustered
        self.config = config
        self.N = len(objs)

        self.org_objs = []
        for i in range(self.N):
            tmp = self._get_combinations(objs[i], i)
            for obj in tmp:
                self.org_objs.append(obj)
        
        self.org_objs = np.array(self.org_objs)
        if self.config.verbose >=1:
            print("[StrokeClustering] info: total {0} combinations were added".format(len(self.org_objs)))

    
    @classmethod
    def fromDir(cls, dir, config, n=None):
        """Explore the directory and cluster all the sketches inside the directory

        Args:
            dir (str): a directoy where the sketches exits
        """
        objs = [] # sketches will be stored in the format of a single object
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
        
        return cls(objs, config)

    def _get_combinations(self, obj:UnlabeledObject, id):
        """for a given object, obtain all possible combinations of strokes

        Args:
            obj ([UnlabeledObject]): [description]
            id (int): unique id to label the combination of a specific sketch
            mx_dis (int): maximum distance between any two strokes TODO: experiment removing this parameter

        Returns: list of dict, each of have information of obj, id, l, r
        """
        stroke_lst = obj.get_strokes()
        res = []
        for i in range(len(stroke_lst)):
            tmp, total_len, last_obj = [], 0, None
            for j in range(i, len(stroke_lst)):
                if total_len > self.config.mx_len or (j != i and self._objs_dist(UnlabeledObject([stroke_lst[j-1]]), UnlabeledObject([stroke_lst[j]])) > self.config.mx_dis):
                    break
                tmp.append(stroke_lst[j].get_copy())
                total_len += len(stroke_lst[j])
                last_obj = UnlabeledObject(copy.deepcopy(tmp))
                res.append({"obj": last_obj, "id":id, "l":i, "r":j, "len":(j-i+1)}) # let represent the number of strokes
        
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

    def _visualize_TSNE(self, embeddings, labels, false_neg=[], false_pos=[], save_path=None):
        # obtain tsne  embeddings
        tsne_embeddings = TSNE(2).fit_transform(embeddings)
        self.cluster_colors = []
        n_clusters = max(labels)

        plt.figure()
        plt.plot(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 'k.')
        # color the clusters
        for i in range(n_clusters):
            tmp = np.where(labels==i)[0]
            # plotting the cluster with TSNE 
            plt.plot(tsne_embeddings[tmp, 0], tsne_embeddings[tmp, 1], '.')
            # append cluster color
            self.cluster_colors.append(plt.gca().lines[-1].get_color())

        # color false negative with green boarder 
        if len(false_neg) > 0:
            plt.scatter(tsne_embeddings[false_neg, 0], tsne_embeddings[false_neg, 1], marker = 'o', c='green', s=5**2, alpha=0.6)
        
        # color false negative with fusia boarder
        if len(false_pos) > 0:
            plt.scatter(tsne_embeddings[false_pos, 0], tsne_embeddings[false_pos, 1], marker = 'o', c='fuchsia', s=5**2, alpha=0.6)

        if self.config.verbose >= 4 and save_path != None:
            plt.savefig(save_path)
        
        if self.config.verbose >= 5:
            plt.show()
    
    def _visualize_samples(self, objs, save_path=None):
        """visualize given objects

        Args:
            objs ([type]): [description]
            save_path ([type], optional): [description]. Defaults to None.
        """
        dim = int(np.sqrt(len(objs)))
        fig, axs = plt.subplots(dim, dim)  
        for i in range(dim):
            for j in range(dim):
                objs[i*dim+j].visualize(ax=axs[i, j], show=False)
                axs[i, j].set_axis_off()
        
        if self.config.verbose >= 4 and save_path != None:
            plt.savefig(save_path)
        
        if self.config.verbose >= 5:
            plt.show()

    def _visualize_random_samples(self, labels, save_path=None):
        """Visualizing reandom samples from each cluster
        """ 
        n_clusters = max(labels)
        fig, axs = plt.subplots(n_clusters, self.config.num_vis_samples_per_cluster+1) 
        for i in range(n_clusters):
            cluster = np.where(labels==i)[0]

            # plot color for the first cell
            axs[i, 0].plot([0], [0], 'o', color=self.cluster_colors[i], markersize=25)
            axs[i, 0].set_axis_off()
            axs[i, 0].set_title(f"{str(len(cluster))}" , loc='left')
            
            inds = rnd.choices(range(len(cluster)), k=self.config.num_vis_samples_per_cluster)
            for j, ind in enumerate(inds):
                self.org_objs[cluster[ind]]['obj'].visualize(ax=axs[i, j+1], show=False)
                axs[i, j+1].set_axis_off()

        if self.config.verbose >= 4 and save_path != None:
            plt.savefig(save_path)
        
        if self.config.verbose >= 5:
            plt.show()

    def _filter(self, lst, indcies):
        """
        remove elements of given indicies form the list
        """
        return np.asarray([obj for i, obj in enumerate(lst) if i not in indcies])

    def _is_intersected(self, obj1, check):
        """
        True if check is intersected with obj1 but obj1 is not fully contained within check 
        """
        if obj1 == check or obj1['id'] != check['id']:
            return False

        l1, r1, l2, r2 = obj1['l'], obj1['r'], check['l'], check['r']

        #            before        after                check is fully contained in obj1
        return not ((r1 < l2) or (l1 > r2) or self._is_contained(check, obj1)) #TODO check this out, this would remove check that fully contains object CHECK

    def _eliminate_intersected(self, obj):
        indices = [i for i, a in enumerate(self.org_objs) if self._is_intersected(obj, a)]
        self.org_objs = self._filter(self.org_objs, indices)
        self.org_embds = self._filter(self.org_embds, indices)

    def _is_contained(self, obj1, check): # TODO contained is already covered with intersected CHECK
        """
        True if check is a part of obj1
        """
        if obj1 == check or obj1['id'] != check['id']:
            return False

        l1, r1, l2, r2 = obj1['l'], obj1['r'], check['l'], check['r']
        return l2 >= l1 and r2 <= r1

    def _eliminate_contained(self, obj):
        indices = [i for i, a in enumerate(self.org_objs) if self._is_contained(obj, a)]
        self.org_objs = self._filter(self.org_objs, indices)
        self.org_embds = self._filter(self.org_embds, indices)

    def _find_false_negative(self, pred_objs, true_objs, save_path=None):
        """Find objects in true_objs that are not in pred_objs and return their indicies in the self.org_objs

        Args:
            pred_objs ([type]): [description]
            true_objs ([type]): [description]
            save_path ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        indices, tot = [], 0
        obj_to_vis = []
        for obj in true_objs:
            if obj not in pred_objs:
                obj_to_vis.append(obj)
                # search for the index in org_objs
                
                ind = -1
                for i, tpl in enumerate(self.org_objs):
                    if obj == tpl['obj']:
                        ind = i
                if ind == -1:
                    tot += 1
                else:
                    indices.append(ind)
        print("[StrokeClustering] info: total {0} object not found in combinations. Might be eliminated as a part of other selected object".format(tot))
        if len(obj_to_vis) > 0:
            self._visualize_samples(obj_to_vis[:self.config.num_vis_samples], save_path=save_path)
        return indices, len(indices) + tot, tot
    
    def _find_false_positive(self, pred_objs, true_objs, save_path=None):
        indices, tot = [], 0
        obj_to_vis = []
        for obj in pred_objs:
            if obj not in true_objs:
                # search for the index in org_objs
                obj_to_vis.append(obj)
                ind = -1
                for i, tpl in enumerate(self.org_objs):
                    if obj == tpl['obj']:
                        ind = i
                if ind == -1:
                    tot += 1
                    print("[StrokeClustering] error: object not found in combinations although it was selected")
                else:
                    indices.append(ind)
        if len(obj_to_vis) > 0:
            self._visualize_samples(obj_to_vis[:self.config.num_vis_samples], save_path=save_path)
        return indices, len(indices) + tot, tot

    def segment(self, true_objs=None):
        fp_lst, fn_lst, tp_lst, num_objs_discarded = [], [], [], []
        precision_lst, recall_lst, wrong_eliminated_objs = [], [], []
        mn_samples, eps = self.config.mnPts, self.config.eps
        if mn_samples <= 1:
            mn_samples = int(self.N * mn_samples)
        
        # logging
        if self.config.verbose:
            print("[StrokeClustering] info: eps initial value:{0}".format(eps))
            print("[StrokeClustering] info: mn_samples:{0}".format(mn_samples))

        # obtain embeddings
        self.org_embds = ObjectUtil.get_embedding([obj['obj'] for obj in self.org_objs])
        if self.config.verbose:
            print("original embeddings obtained")

        for d in range(self.config.iterations):
            labels = np.asarray(DBSCAN(eps=eps + d * self.config.eps_inc_rate, min_samples=mn_samples).fit(self.org_embds).labels_)
            pred_objs = [tpl['obj'] for tpl in self.org_objs[labels != -1]]
            pred_segs = [(tpl['l'], tpl['r']) for tpl in self.org_objs[labels != -1]]

            if true_objs is not None:
                fn_ind, fn, tot = self._find_false_negative(pred_objs, true_objs, save_path=os.path.join(self.config.fn_fig_dir, f'iter_{d}.png'))    
                fp_ind, fp, _ = self._find_false_positive(pred_objs, true_objs, save_path=os.path.join(self.config.fp_fig_dir, f'iter_{d}.png'))
                tp = len(pred_objs) - fp
                
                if self.config.verbose >= 1:
                    print("[StokeClustering] info: iter {1} - total {0} false positive".format(fp, d))
                    print("[StokeClustering] info: iter {1} - total {0} false negative".format(fn, d))
                    print("[StokeClustering] info: iter {1} total {0} true positive".format(tp, d))
                    print("[StokeClustering] info: iter {1} precision:{0}".format((tp)/(tp+fp), d))
                    print("[StokeCllsustering] info: iter {1} recall:{0}".format((tp)/(tp+fn), d))
                    fp_lst.append(fp)
                    fn_lst.append(fn)
                    tp_lst.append(tp)
                    precision_lst.append((tp)/(tp+fp))
                    recall_lst.append((tp)/(tp+fn))
                    wrong_eliminated_objs.append(tot)



            if self.config.verbose >= 4:
                self._visualize_TSNE(self.org_embds, labels, false_neg=fn_ind, false_pos=fp_ind, save_path=os.path.join(self.config.tsne_fig_dir, f'iter_{d}.png'))
                self._visualize_random_samples(labels, save_path=os.path.join(self.config.clusters_fig_dir, f'iter_{d}.png'))


            n_clusters = max(labels)
            clusters = [self.org_objs[np.where(labels==i)[0]] for i in range(n_clusters)]

            pre_len = len(self.org_objs)
            for cluster in clusters:
                # if np.any([obj['len'] == 1 for obj in cluster]): # check for single stroke obj
                for obj in cluster:
                    if obj['len'] == 1:
                        continue # objects of length one or already processed objects do not need to be processed again 
                    self._eliminate_contained(obj) # TODO: we are only eliminating the contained objects 
                    # obj['len'] = 1 # edit len to one. This is used in order to not process the project again # TODO: check this assumption

            if self.config.verbose >= 1:
                print("[StrokeClustering] info: {0} objects were discarded".format(pre_len-len(self.org_objs)))
                print("-------------------------------------------------------------")
                num_objs_discarded.append(pre_len-len(self.org_objs))

        if self.config.verbose >= 3:
            # preapre summary figures
            # tp, tn, fp
            fig, ax = plt.subplots()
            ax.plot(tp_lst, label='TP')
            ax.plot(fp_lst, label='FP')
            ax.plot(fn_lst, label='FN')
            ax.plot(num_objs_discarded, label='Number of objects discarded')
            ax.plot(wrong_eliminated_objs, label='Wrong eliminated objects')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Number of Objects')
            fig.legend()
            fig.savefig(os.path.join(self.config.exp_dir, 'objects_analysis.png'))

            # precision recall
            fig, ax = plt.subplots()
            ax.plot(precision_lst, label='Precision')
            ax.plot(recall_lst, label='Recall')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Value')
            fig.legend()
            fig.savefig(os.path.join(self.config.exp_dir, 'precision_recall.png'))

        # return found objects
        return pred_objs, pred_segs


        

        
