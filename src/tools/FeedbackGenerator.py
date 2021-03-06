from registrationNN.models import model_visualizer
from utils.ObjectUtil import ObjectUtil
import numpy as np
from utils.RegistrationUtils import RegistrationUtils
from utils.ObjectUtil import ObjectUtil
import numpy as np
from lapjv import lapjv
from utils.RegistrationUtils import RegistrationUtils
from sketch_object.UnlabeledObject import UnlabeledObject
import copy
from sketch_object.Stroke import Stroke
from animator.SketchAnimation import SketchAnimation
import pickle5 as pickle
import os

class VideoGenerator:
    def __init__(self, alignment_model, config):
        self.config = config

        # load pretrained model weights
        self.alignment_model = alignment_model
        
    
    def generate(self, org_sketch, tar_sketch):
        """given two lists of objects, generate feedback video on how to go from one sketch to another

        Args:
            org_sketch (List(UnlabeledObject)): Objects or original sketch
            tar_sketch (List(UnlabeledObject)): Objects or target sketch
        """
        n, m = len(org_sketch), len(tar_sketch)

        if self.config.load_trans_params and os.path.exists(os.path.join(self.config.video_dir, 'transformation_parameters.pkl')):
            with open(os.path.join(self.config.video_dir, 'transformation_parameters.pkl'), 'rb') as f:
                final_params = pickle.load(f)
            
            if self.config.verbose > 2:
                print(f'[VideoGenerator] info: final transformation params: {final_params}')
    
        else:
            # prepare pair-wise set
            org_objs, tar_objs = [], []
            for obj1 in org_sketch:
                for obj2 in tar_sketch:
                    org_objs.append(obj1)
                    tar_objs.append(obj2)


            self.alignment_model.fine_tune(org_objs, tar_objs, self.config.fine_tune_epochs)

            trans_params, losses = self.alignment_model.predict(org_objs, tar_objs) # trans_params(N * M, 7)
            trans_params = np.reshape(trans_params, (n, m, 7))
            losses = np.reshape(losses, (n, m))

            final_params = self.optimal_transformation(org_sketch, tar_sketch, losses, trans_params) # note: new objects might be added to org_sketch
            if self.config.verbose > 2:
                print(f'[VideoGenerator] info: pair-wise transformation parameters: {trans_params}')
                print(f'[VideoGenerator] info: pair-wise disimilarity: {losses}')
                print(f'[VideoGenerator] info: final transformation params: {final_params}')

            # save final params in a pickle file 
            with open(os.path.join(self.config.video_dir, 'transformation_parameters.pkl'), 'wb') as f:
                pickle.dump(final_params, f)
       
        # fix and object in the target objects
        cnt_ind = self.get_widest_obj_ind(tar_sketch)
        
        # normalize all transformation according to the transformation of the fixed object
        for i in range(len(final_params)):
            final_params[i][5] -= final_params[cnt_ind][5]
            final_params[i][6] -= final_params[cnt_ind][6]

        if self.config.verbose > 3:
            model_visualizer.visualize_model(self.alignment_model, org_sketch, tar_sketch, org_sketch, tar_sketch, self.alignment_model.model_config)


        if self.config.vis_video:
            # generate the video based on the final params
            anim = SketchAnimation(org_sketch, tar_sketch)
            anim.seq_animate_all(final_params, save=True, file=self.config.save_video_path, denormalize_trans=True)
        

    def get_widest_obj_ind(self, objs):
        mx_width, ind = 0, 0
        for i, obj in enumerate(objs):
            width = max(obj.get_x()) - min(obj.get_x())
            if width > mx_width:
                mx_width = width
                ind = i
        return ind


    def optimal_transformation(self, org_objs, tar_objs, dissimilarity_matrix, trans_matrix):
        """Based on the optimal assignment solution, ditribute the transformation parameters for each original object.

        Args:
            org_objs: list of the original objects in the sketch (mutable: new object might be added)
            tar_objs: list of the target objects in the sketch
            dissimilarity_matrix (array(N,M)): pair-wise disimilarity
            trans_matrix (array(N,M)): [description]

        Returns:
            M x 7: transformation parameters for each object in the original sketch
        """
        
        n, m = len(org_objs), len(tar_objs)
        dim = max(n,m)
        self.res_matrix = np.zeros((dim, dim))
        self.tra_matrix = np.zeros((dim, dim, 7))  

        # fill the result in the res_matrix
        for i in range(max(n,m)):
            for j in range(max(n,m)):
                if i >= n or j >= m:
                    d, p = RegistrationUtils.inf, np.zeros(7)
                else:
                    d, p = dissimilarity_matrix[i][j], trans_matrix[i][j]

                self.res_matrix[i, j] = d
                self.tra_matrix[i, j] = p

        # calculate the minimum assignment
        org_asg, tar_asg, total_cost = lapjv(self.res_matrix)

        print("Original assignment:", org_asg)
        print("Matched objects losses:", [self.res_matrix[i][asig] for i, asig in enumerate(org_asg)])
        print("Total alignment cost:", total_cost)

        final_transformation = np.zeros((n, 7))
        added_objects = []

        for i, ind in enumerate(org_asg):
            dissimilarity = self.res_matrix[i, ind]
            # if i < n and ind < m:
            # # calculate the disimilarity again after restructing the object TODO: delete
            #     ln = max(len(org_objs[i]), len(tar_objs[ind]))
            #     ref_obj = ObjectUtil.object_restructure(org_objs[i], n=ln)
            #     tar_obj = ObjectUtil.object_restructure(tar_objs[ind], n=ln)
            #     dissimilarity = RegistrationUtils.calc_dissimilarity(ref_obj, tar_obj, trans_matrix[i, ind], cum_ang=False, turning_ang=False)
            # print(dissimilarity, self.res_matrix[i, ind])
            
            # check if one of the objects is recently added (dummy) or their dissimilarity is above the maximum threshold
            if dissimilarity > self.config.mx_dissimilarity:
                non_added_object = dissimilarity != RegistrationUtils.inf
                
                if non_added_object and self.config.verbose > 1:
                    print(f'[VideoGenerator] info: Original object {i} could not find a good match. Lowest dissimilarity: {dissimilarity}')

                # handle the case when n > m or when the object does not have any good match
                # by making the object vanish into its origin
                if n > m or non_added_object:
                    if self.config.verbose > 1:
                        print(f'[VideoGenerator] info: Original object {i} will vanish as it could not find any match')
                    self.tra_matrix[i, ind] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, org_objs[i].origin_x, org_objs[i].origin_y])        
                
                # handle the case when m > n or when the object does not have any good match
                # by creating a new object in the orginal sketch, identical to the target sketch but scalled by a very small scale
                if m > n or non_added_object:
                    stroke_lst = copy.deepcopy(tar_objs[ind].get_strokes())
                    new_obj = UnlabeledObject(stroke_lst)
                    eps = self.config.construction_step_size

                    # create an identical object at a small scale
                    new_obj_stroke_lst = []
                    for st in new_obj.get_strokes():
                        for pt in st.get_points():
                            pt.x = pt.x * eps + (1 - eps) * new_obj.origin_x
                            pt.y = pt.y * eps + (1 - eps) * new_obj.origin_y
                        new_obj_stroke_lst.append(Stroke(st.get_points()))
                    new_obj = UnlabeledObject(new_obj_stroke_lst)
                    
                    final_transformation = np.append(final_transformation, np.array([[1/eps, 1/eps, 0.0, 0.0, 0.0, 0.0, 0.0]]), axis=0)

                    org_objs.append(new_obj)
                    added_objects.append(ind)

            if i < n:
                final_transformation[i] = self.tra_matrix[i, ind]

        return final_transformation

           
    def spatial_redistribution(self, target_groups, org_asg, tar_asg):
        """redistribute the assignment of the identical object by considering their spatial relation 
        
        Parameters:
            target_groups: the indecies of orginal object grouped
            org_asg: the initial assignment of the orginal objects
            tar_asg: the initial assignment of the target objects

        Returns:
            None. mutate org_asg and tar_asg
        """  
        for group in target_groups:
            n = len(group)
            if n == 1:
                continue
            org = []
            for obj_ind in group:
                org.append(tar_asg[obj_ind])
            weight_matrix = np.zeros((n, n))
            for i in org:
                for j in group:
                    weight_matrix[i][j] = self.tra_matrix[i][j][5] ** 2 + self.tra_matrix[i][j][6] ** 2
            row_ind, col_ind, _ = lapjv(weight_matrix)
            for i, ind in enumerate(row_ind):
                org_asg[org[i]] = group[ind]
            for i, ind in enumerate(col_ind):
                tar_asg[group[i]] = org[ind]

       