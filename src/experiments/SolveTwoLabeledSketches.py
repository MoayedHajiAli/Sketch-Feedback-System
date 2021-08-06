import sys
sys.path.insert(0, '../')

from utils.Config import Config
from utils.ObjectUtil import ObjectUtil
from tools.FeedbackGenerator import VideoGenerator
from registrationNN.models import NNModel
import time


if __name__ == '__main__':
    print(f'[SolveTwoLabeledSketches] {time.ctime()}: Started generating feedback video')
    model_tag = 'after-decomposition-penalty' # deep learning model to use for alignment
    question_name = 'ReflectionQuestion'
    org_sketch_id = '2_58c52b2b-94f4-49e1-b94c-d93964b1319c'
    tar_sketch_id = '2_a54ad2f8-3660-402e-a00a-d199e4679d39'

    config = Config.default_video_config(question_name, org_sketch_id, tar_sketch_id)
    config.vis_video = True
    config.load_trans_params = False
    config.fine_tune_epochs = 50
    config.verbose = 4

    model_params = Config.default_model_config(model_tag)
    model_params.load = False
    model_params.learning_rate = 1e-3
    
    # load original and target sketch 
    org_sketch, org_labels = ObjectUtil.xml_to_UnlabeledObjects(config.org_sketch_path, 
                                                                mn_len=config.mn_len, 
                                                                re_sampling=config.re_sampling, 
                                                                flip=config.org_flip, 
                                                                shift_x=config.org_shift_x,
                                                                 shift_y=config.org_shift_y)

    tar_sketch, tar_labels = ObjectUtil.xml_to_UnlabeledObjects(config.tar_sketch_path, 
                                                                mn_len=config.mn_len, 
                                                                re_sampling=config.re_sampling, 
                                                                flip=config.tar_flip, 
                                                                shift_x=config.tar_shift_x,
                                                                 shift_y=config.tar_shift_y)


    generator = VideoGenerator(NNModel(model_params), config)
    generator.generate(org_sketch, tar_sketch)
    print(f'[SolveTwoLabeledSketches] {time.ctime()}: finished generating video')
    