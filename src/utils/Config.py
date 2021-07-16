import os
from munch import Munch


class Config:

    def default_model_config(exp_id):
        # prepare configuration
        config = Munch()
        config.exp_id = exp_id

        # dataset realted
        config.n_file = 5000
        config.k_select = 200
        config.obj_accepted_labels = ['Circle', 'Star', 'Triangle', 'Star Bullet', 'Square', 'Arrow Right', 'Trapezoid Down', 'Trapezoid Up', 'Diamond', 'Square', 'Plus', 'Upsidedown Triangle', 'Minus']
        config.dataset_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_A')
        config.seed = 1

        # model related
        config.batch_size = 128
        config.learning_rate = 2e-3
        config.epochs = 200
        config.load = False
        config.load_ckpt = False
        config.save = True
        config.save_ckpt = True
        config.save_best_only = True
        config.exp_dir = "../registrationNN/saved_models/experiment{0}".format(str(exp_id)) # save path
        config.hist_path = config.exp_dir + "/hist.pkl"
        config.ckpt_path = config.exp_dir + "/cp-{epoch:04d}.ckpt"
        config.verbose = 5

        # visualization realted
        config.vis_train = True
        config.vis_test = True
        config.fine_tune = True
        config.fine_tune_epochs = 10
        config.iter_refine_prediction = True
        config.vis_transformation = False
        config.vis_dir = "../registrationNN/saved_results/experiment{0}".format(str(exp_id))
        config.num_vis_samples = 10

        os.makedirs(config.exp_dir, exist_ok=True)
        os.makedirs(config.vis_dir, exist_ok=True)

        return config

    def default_segmentation_config(exp_id):
        # prepare configuration
        config = Munch()
        config.exp_id = exp_id

        # experiment realted
        config.n_file = 5000
        config.remove_lables = ['Number']
        config.test_dir = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_A/MoneyQuestion')
        config.tar_file = None
        config.seed = 1
        config.verbose = 5
        config.re_sampling = 0.0
        

        # algorith related
        config.eps = 8
        config.mnPts = 0.1
        config.mx_dis = 30
        

        # visualization realted
        

        return config