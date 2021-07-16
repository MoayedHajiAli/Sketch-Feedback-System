
import sys
sys.path.insert(0, '../')

from tools.ParsingEvaluation import ParsingEvaluation
import os.path as path
from utils.Config import Config

exp_config = exp_config.default_segmentation_config()
evaluator = ParsingEvaluation(str(test_dir), str(tar_obj_path), n_files=-1)
evaluator.evaluate()

