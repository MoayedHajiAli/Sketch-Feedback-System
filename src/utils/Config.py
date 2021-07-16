import os
from munch import Munch


class Config:

    def default_model_config(exp_id):
        # prepare configuration
        model_config = Munch()
        model_config.exp_id = exp_id

        # dataset realted
        model_config.n_file = 5000
        model_config.k_select = 200
        model_config.obj_accepted_labels = ['Circle', 'Star', 'Triangle', 'Star Bullet', 'Square', 'Arrow Right', 'Trapezoid Down', 'Trapezoid Up', 'Diamond', 'Square', 'Plus', 'Upsidedown Triangle', 'Minus']
        model_config.dataset_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_A')
        model_config.seed = 1

        # model related
        model_config.batch_size = 128
        model_config.learning_rate = 2e-3
        model_config.epochs = 200
        model_config.load = False
        model_config.load_ckpt = False
        model_config.save = True
        model_config.save_ckpt = True
        model_config.save_best_only = True
        model_config.exp_dir = "../registrationNN/saved_models/experiment{0}".format(str(exp_id)) # save path
        model_config.hist_path = model_config.exp_dir + "/hist.pkl"
        model_config.ckpt_path = model_config.exp_dir + "/cp-{epoch:04d}.ckpt"
        model_config.verbose = 5

        # visualization realted
        model_config.vis_train = True
        model_config.vis_test = True
        model_config.fine_tune = True
        model_config.fine_tune_epochs = 10
        model_config.iter_refine_prediction = True
        model_config.vis_transformation = False
        model_config.vis_dir = "../registrationNN/saved_results/experiment{0}".format(str(exp_id))
        model_config.num_vis_samples = 10

        os.makedirs(model_config.exp_dir, exist_ok=True)
        os.makedirs(model_config.vis_dir, exist_ok=True)

        return model_config