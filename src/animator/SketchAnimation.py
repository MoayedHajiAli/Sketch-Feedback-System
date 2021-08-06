import time

from utils.ObjectUtil import ObjectUtil
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from utils.RegistrationUtils import RegistrationUtils
import copy


class SketchAnimation():
    original_labels = []
    target_labels = []
    dim = [[-500, 2000], [-500, 2000]]
    
    def __init__(self, original_obj, target_obj):
        self.original_obj = original_obj
        self.target_obj = target_obj

        # take deep copies 
        for i in range(len(self.original_obj)):
            self.original_obj[i] = self.original_obj[i].get_copy()
            self.target_obj[i] = self.target_obj[i].get_copy()
        
        # figure
        self.fig, self.ax = plt.subplots()

        # create plot patches for every stroke
        self.original_patches = []
        for obj in self.original_obj:
            self.original_patches.append([self.ax.plot([], [])[0] for _ in range(len(obj.get_strokes()))])

        self.target_patches = []
        for obj in self.target_obj:
            self.target_patches.append([self.ax.plot([], [])[0] for _ in range(len(obj.get_strokes()))])

        
    # update the coordinates of the all the objects
    # lst: list of objects, x, y: list of x, y traverse for objects
    def move_all(self, lst, steps):
        for obj in lst:
            obj.move_step(steps)

    # for a given two objects, match the object by trying all pair of points
    # and calculate the x, y traverse
    def find_single_translation_path(self, obj1, obj2):
        # calculate start and end position
        mn_RMS, stx, sty, enx, eny = ObjectUtil.find_optimal_translation(obj1, obj2)
        return (enx - stx), (eny - sty)

    # for a given two set of objects, calculate the x, y traverse for every step
    # returns a list of X, Y
    def find_translation_paths(self, org_objs, tar_objs):
        X, Y = [], []
        for obj1, obj2 in zip(org_objs, tar_objs):
            x, y = self.find_single_translation_path(obj1, obj2)
            X.append(x)
            Y.append(y)
        return X, Y

    # return all strokes' patches
    def get_all_patches(self):
        all_patches = [pat for lst in self.original_patches for pat in lst]
        all_patches.extend([pat for lst in self.target_patches for pat in lst])
        return all_patches

    # initial function for animation
    def _init_animation(self):
        self.ax.set_xlim(self.dim[0][0], self.dim[0][1])
        self.ax.set_ylim(self.dim[1][0], self.dim[1][1])

        # fix original video origin to 300, 300
        org_x, org_y = 1e6, 1e6
        for obj in self.original_obj:
            org_x = min(org_x, obj.origin_x)
            org_y = min(org_y, obj.origin_y)

        self.org_shift = (-(org_x - 300), -(org_y - 300))
        for obj in self.original_obj:
            obj.reset()
            obj.transform([1.0, 0.0, self.org_shift[0], 0.0, 1.0, self.org_shift[1]], upd_step=False)
            obj.origin_min_x += self.org_shift[0]
            obj.origin_min_y += self.org_shift[1]

        for pt_lst, obj in zip(self.original_patches, self.original_obj):
            for pt, stroke in zip(pt_lst, obj.get_strokes()):
                pt.set_data(stroke.get_x(), stroke.get_y())

        # fix target origin to 1000, 1000
        org_x, org_y = 1e6, 1e6
        for obj in self.target_obj:
            org_x = min(org_x, obj.origin_x)
            org_y = min(org_y, obj.origin_y)

        self.tar_shift = (-(org_x - 1000), -(org_y - 1000))
        for obj in self.target_obj:
            obj.reset()
            obj.transform([1.0, 0.0, self.tar_shift[0], 0.0, 1.0, self.tar_shift[1]], upd_step=False)
            obj.origin_min_x += self.tar_shift[0]
            obj.origin_min_y += self.tar_shift[1]

        for pt_lst, obj in zip(self.target_patches, self.target_obj):
            for pt, stroke in zip(pt_lst, obj.get_strokes()):
                pt.set_data(stroke.get_x(), stroke.get_y())

        return self.get_all_patches()

    # animate function for matplotlib animation. It is being called in every step
    def _animate(self, i, steps):
        self.move_all(self.original_obj, steps)

        for pt_lst, obj in zip(self.original_patches, self.original_obj):
            for pt, stroke in zip(pt_lst, obj.get_strokes()):
                pt.set_data(stroke.get_x(), stroke.get_y())

        return self.get_all_patches()

    def animate_translation_all(self, steps=1000):
        tem_org, tem_tar = [], []
        leb_org, leb_tar = [], []
        x_tra, y_tra = [], []
        n = max(self.original_labels)
        for i in range(n + 1):
            tem_org.extend(self.original_obj[self.original_labels == i])
            tem_tar.extend(self.target_obj[self.target_labels == i])
            tem_x, tem_y = self.find_translation_paths(self.original_obj[self.original_labels == i],
                                                       self.target_obj[self.target_labels == i])
            x_tra.extend(tem_x)
            y_tra.extend(tem_y)

        # update order of objects of the original and target
        self.original_obj, self.target_obj = tem_org, tem_tar

        # update labels
        self.original_labels, self.target_labels = leb_org, leb_tar

        # translate objects to obtain SketchAnimationing step vectors then reset
        for obj, x, y, in zip(self.original_obj, x_tra, y_tra):
            obj.transform([1, 0, x, 0, 1, y])
            obj.reset()

        self.animate(steps=steps)

    def animate(self, steps=1000, save=False, file="example3.mp4"):
        # animate
        anim = animation.FuncAnimation(self.fig, func=self._animate,
                                       init_func=self._init_animation, frames=steps, interval=1, blit=True,
                                       repeat_delay=2000, fargs=[steps], repeat=True)
        if save:
            self._save_anim(anim, file)
        plt.show()
        return anim

    def animate_all(self, trans_matrix, steps=300, save=False, file="example3.mp4"):
        # transform all objects to obtain SketchAnimationing step vectors then reset them to their original coordinates
        for obj, t in zip(self.original_obj, trans_matrix):
            obj.reset()
            obj.upd_step_vector(t)
        self.animate(steps=steps, save=save, file=file)

    def _save_anim(self, anim, name):
        # Set up formatting for the movie files
        anim.save(name, writer='ffmpeg')


    def _seq_anim(self, i, steps, trans_matrix):
        if i % steps == 0:
            # apply the next transformation
            ind = int(i/steps)
            for obj, t in zip(self.original_obj, trans_matrix):
                obj.upd_step_vector(t[ind])
            # time.sleep(0.5) # TODO: check if needed

        # move objects
        self.move_all(self.original_obj, steps)
        for pt_lst, obj in zip(self.original_patches, self.original_obj):
            for pt, stroke in zip(pt_lst, obj.get_strokes()):
                pt.set_data(stroke.get_x(), stroke.get_y())

        return self.get_all_patches()

    def _seq_obj_anim(self, i, obj_steps, steps, trans_matrix):
        obj_ind = int(i/obj_steps)
        if (i % obj_steps) % steps == 0:
            # apply the next transformation
            ind = int((i % obj_steps)/steps)
            print(ind)
            if ind == 4: # translation
                tmp = copy.deepcopy(trans_matrix[obj_ind][ind])

                # tmp[2] += self.tar_shift[0] 
                # tmp[5] += self.tar_shift[0]
                tmp[2] += self.org_shift[0]
                tmp[5] += self.org_shift[1]
                self.original_obj[obj_ind].upd_step_vector(tmp, object_min_origin=True, retain_origin=False)

            else:
                print(trans_matrix[obj_ind][ind])
                self.original_obj[obj_ind].upd_step_vector(trans_matrix[obj_ind][ind], object_min_origin=True, retain_origin=True)
            # time.sleep(0.2) # TODO check if needed

        # move objects
        self.original_obj[obj_ind].move_step(steps)
        for pt_lst, obj in zip(self.original_patches, self.original_obj):
            for pt, stroke in zip(pt_lst, obj.get_strokes()):
                pt.set_data(stroke.get_x(), stroke.get_y())

        return self.get_all_patches()

    # animate according to the transformation parameters p, where p has 7 parameters as follows:
        # p[0]: the scaling the x direction
        # p[1]: the scaling the y direction
        # p[2]: rotation for theta degrees (counter clock-wise in radian)
        # p[3]: shearing in the x axis
        # p[4]: shearing in the y axis
        # p[5]: translation in the x direction
        # p[6]: translation in the y direction
    # the order of transformation is scaling -> shaering -> roation -> translation
    def seq_animate_all(self, transformation_params, steps=200, save=False, file="example4.mp4", denormalize_trans=False):
        # for each object holds 5 sequential transformation matrices: shearing-y -> shearing-x -> rotation -> scaling -> translation
        all_t_params = [] # N x 5

        self.denormalize_trans = denormalize_trans
        # add scaling transformation matrix
        for ind, p in enumerate(transformation_params):
            if denormalize_trans:
                print("Before", p)
                t = RegistrationUtils.obtain_transformation_matrix(p)
                # denormalize
                t_denormalized = ObjectUtil.denormalized_transformation(self.original_obj[ind], self.target_obj[ind], t)
                # decompose
                p_denormalized = RegistrationUtils.decompose_tranformation_matrix(t_denormalized)
                print("After", p_denormalized)

            seq_params = RegistrationUtils.get_seq_translation_matrices(p_denormalized)
            all_t_params.append(seq_params)
            

        print("Seq Params:", all_t_params)
        # animate
        print(len(self.original_obj),  5 * steps)
        anim = animation.FuncAnimation(self.fig, func=self._seq_obj_anim,
                                       init_func=self._init_animation, 
                                       frames=len(self.original_obj) * 5 * steps, # 5 transformation steps for each object
                                       interval=1,
                                       blit=True,
                                       repeat=False,
                                       fargs=[5 * steps, steps, all_t_params])


        plt.show()
        if save:
            self._save_anim(anim, file)

    
        return anim


