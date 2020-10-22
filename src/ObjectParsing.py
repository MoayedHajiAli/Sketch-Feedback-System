from RegistrationUtils import RegisterTwoObjects, RegistrationUtils
from Morph import Morph
from UnlabeledObject import UnlabeledObject
from ObjectUtil import ObjectUtil
import numpy as np
import time
from matplotlib import pyplot as plt

class ObjectParsing:

    @staticmethod
    def extract_corresponding_strokes(ref_obj, tar_obj, p):
        """for a given ref_obj, tar_obj and transformation params for the ref_obj, find all strokes in the tar_obj that matched the 
        transformed object

        Args:
            ref_obj (UnlabeledObject): 
            tar_obj (UnlabeledObject): 
            p (array-like(6)): the values of the transformation matrix
        
        Returns:
            a set of the corresponding strokes
        """
        # transform both object to the origin of the referenced object
        RegistrationUtils.normalize_coords(ref_obj, tar_obj, -1)

        # store transformed points
        x, y = np.array(copy.deepcopy(ref_obj.get_x())), np.array(copy.deepcopy(ref_obj.get_y()))
        x1, y1 = np.array(copy.deepcopy(tar_obj.get_x())), np.array(copy.deepcopy(tar_obj.get_y()))

        # transform both object to the origin of the referenced object
        RegistrationUtils.normalize_coords(ref_obj, tar_obj, 1)

        # transform the org_obj
        x, y = RegistrationUtils.transform(x, y, p)

        target_nn = Nearest_search(x1, y1)

        strokes = set([])
        for i in range(len(x)):
            ind = target_nn.query_ind(x[i], y[i])
            strokes.add(ind)
        
        return strokes

    @staticmethod
    def find_matched_strokes(strokes_lst, obj:UnlabeledObject, acceptance_threshold, total_cost):
        matched_collections, costs, params = [], [], []
        # make a new object out of all given strokes (i.e out of the enitre sketch)
        sketch = UnlabeledObject(strokes_lst)
        visited_strokes = set([])
        registration = RegisterTwoObjects(sketch, obj, total_cost)
        for i, stroke in enumerate(strokes_lst):
            # check if this stroke is visited
            if i in visited_strokes:
                continue
            # change sketch origin to match the stroke origin
            sketch.reset()
            obj.reset()
            sketch.origin_x = stroke.origin_x
            sketch.origin_y = stroke.origin_y 
            print("Starting positiolns:" , sketch.origin_x , sketch.origin_y)
            obj1, obj2 = obj, sketch
            st = time.time()
            d, p = RegisterTwoObjects(obj1, obj2, RegistrationUtils.total_transformation_cost).optimize(target_dis=False)
            print(d, [np.array(p)])
            print("Running time: ", time.time()-st)
            # print(RegistrationUtils.identify_similarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p)))
            morph = Morph([obj1], [obj2])
            morph.seq_animate_all([p], save=False, file="./test_videos/example7-obj3-4.mp4")
            plt.show()
            continue
            print("Here1")
            d, p = registration.optimize(target_dis=False)
            print("Here2")
            correspondences = ObjectParsing.extract_corresponding_strokes(obj, sketch, p)
            print("Here3")
            # mark correspondences as visited
            for ind in correspondences:
                visited_strokes.add(ind)
            
            print(d, correspondences)
            if d <= acceptance_threshold:
                matched_collections.append(correspondences)
                costs.append(d)
                params.append(p)
