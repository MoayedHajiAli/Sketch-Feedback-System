import sys
sys.path.insert(0, '../')

from utils.Config import Config
from utils.ObjectUtil import ObjectUtil
from tools.FeedbackGenerator import videoGenerator
from registrationNN.models import NNModel
import time


if __name__ == '__main__':
    print(f'[SolveTwoLabeledSketches] {time.ctime()}: Started generating feedback video')
    model_tag = 'experiment7' # deep learning model to use for alignment
    question_name = 'ReflectionQuestion'
    org_sketch_id = '2_58c52b2b-94f4-49e1-b94c-d93964b1319c'
    tar_sketch_id = '2_78a20c33-66ab-4e7c-9064-30a9372e13c6'

    config = Config.default_video_config(question_name, org_sketch_id, tar_sketch_id)
    model_params = Config.default_model_config(model_tag)
    model_params.load = False
    
    # load original and target sketch 
    org_sketch, org_labels = ObjectUtil.xml_to_UnlabeledObjects(config.org_sketch_path, mn_len=config.mn_len, re_sampling=config.re_sampling)
    tar_sketch, tar_labels = ObjectUtil.xml_to_UnlabeledObjects(config.tar_sketch_path, mn_len=config.mn_len, re_sampling=config.re_sampling)


    generator = videoGenerator(NNModel(model_params), config)
    generator.generate(org_sketch, tar_sketch)
    print(f'[SolveTwoLabeledSketches] {time.ctime()}: finished generating video')
    