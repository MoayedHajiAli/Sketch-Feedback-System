
from importlib import import_module
import sys
sys.path.insert(0, '../')

from tools.ParsingEvaluation import ParsingEvaluation
from utils.Config import Config
import json
import os


# redirect output to log
# sys.stdout = open(os.path.join(save_dir, 'log.out'), 'w+')
# import os
# #evaluate sketch object level segmentation
# test_dir = 'ASIST_Dataset/Data/Data_A/BalanceQuestion'
# tmp_dir = '../input_directory/prototypes'
# tar_obj_path = 'ASIST_Dataset/Data/Data_A/BoxPointerMergedQuestion/8_2ce97cdb-c300-4b52-a9bf-dc3a1bde7ed7.xml'
# # tar_obj_path = 'ASIST_Dataset/Data/Data_A/MoneyQuestion/1_5777f61a-1f9a-45a8-a9aa-7fcd30c8c09a.xml'
# test_dir = path.join(path.abspath(path.join(__file__ ,"../../..")), test_dir)
# tar_obj_path = path.join(path.abspath(path.join(__file__ ,"../../..")), tar_obj_path)

# experiment_id = 8
# save_dir = '../results_log/segmentation/iterative/experiment{0}'.format(experiment_id)
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)






exp_config = Config.default_segmentation_config(exp_id=14)
exp_config.iterations = 10
exp_config.mx_dis = 1000

# redirect output to log
# sys.stdout = open(os.path.join(model_config.exp_dir, 'log.out'), 'w+')

# save experiment configurations
config_json = json.dumps(dict(exp_config), indent=4)
with open(os.path.join(exp_config.exp_dir, 'config.txt'), 'w') as f:
    f.write(config_json)

evaluator = ParsingEvaluation(exp_config)
evaluator.evaluate()


