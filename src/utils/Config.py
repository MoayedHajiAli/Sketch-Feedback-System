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
        config.re_sampling = 128 # < 1 it downsamples according to re_sampling * len, =1 disable, > 1 resample to fixed len
        # config.obj_accepted_labels = ['Circle', 'Star', 'Triangle', 'Star Bullet', 'Square', 'Arrow Right', 'Trapezoid Down', 'Trapezoid Up', 'Diamond', 'Square', 'Plus', 'Upsidedown Triangle', 'Minus']
        config.obj_accepted_labels = ['Circle', 'Triangle', 'Square', 'Trapezoid Down', 'Trapezoid Up', 'Diamond', 'Square', 'Plus', 'Upsidedown Triangle', 'Minus']
        config.dataset_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_A/ReflectionQuestion')
        config.seed = 1

        # model related
        config.scaling_f = 2
        config.shearing_f = 2
        config.rotation_f = 1.0
        config.batch_size = 128
        config.learning_rate = 5e-5
        config.epochs = 200
        config.load = False
        config.load_ckpt = False
        config.save = True
        config.save_ckpt = True
        config.save_best_only = True
        config.exp_dir = "../registrationNN/saved_models/{0}".format(str(exp_id)) # save path
        config.ckpt_path = config.exp_dir + "/cp-best-loss.ckpt"
        config.verbose = 5

        # visualization realted
        config.vis_train = True
        config.vis_test = True
        config.fine_tune = True
        config.fine_tune_epochs = 100
        config.iter_refine_prediction = True
        config.vis_transformation = False
        config.vis_dir = "../registrationNN/saved_results/{0}".format(str(exp_id))
        config.hist_path = config.vis_dir + "/hist.pkl"
        config.config_path = config.vis_dir + "/config.txt"
        config.log_path = config.vis_dir + "/log.txt"
        config.num_vis_samples = 10

        os.makedirs(config.exp_dir, exist_ok=True)
        os.makedirs(config.vis_dir, exist_ok=True)

        return config

    def default_segmentation_config(exp_id):
        # prepare configuration
        config = Munch()
        config.exp_id = exp_id

        # experiment realted
        config.n_files = 100
        config.accepted_labels = ['Arrow Up', 'Arrow Right', 'Arrow Left', 'Two Boxes', 'Two Boxes Null', "Star Bullet",
            'Arrow Down', 'Star', 'Triangle', 'Circle', 'Diamond', 'Square', 'Plus', 'Upsidedown Triangle', 'Minus']
        config.test_dir = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), 'ASIST_Dataset/Data/Data_A/MoneyQuestion')
        config.tar_file = None
        config.seed = 1
        config.verbose = 4 # 1-3 for normal messages, 4 to save visualizations and 5 to show visualization
        config.re_sampling = 0.0
        config.exp_dir = '../results_log/segmentation/iterative/{0}'.format(exp_id)

        # algorithm related
        config.iterations = 60
        config.eps = 8
        config.mnPts = 0.1
        config.eps_inc_rate = 0.1
        config.mx_dis = 30
        config.mx_len = 200 # maximum number of points possible in a single object
        

        # visualization realted
        config.fp_fig_dir = os.path.join(config.exp_dir, 'fp')
        config.fn_fig_dir = os.path.join(config.exp_dir, 'fn')
        config.clusters_fig_dir = os.path.join(config.exp_dir, 'clusters_samples')
        config.tsne_fig_dir = os.path.join(config.exp_dir, 'TSNE')
        config.num_vis_samples = 15
        config.num_vis_samples_per_cluster = 5
        
        os.makedirs(config.fn_fig_dir, exist_ok=True)
        os.makedirs(config.fp_fig_dir, exist_ok=True)
        os.makedirs(config.clusters_fig_dir, exist_ok=True)
        os.makedirs(config.tsne_fig_dir, exist_ok=True)

        return config


    def default_video_config(question_name, sketch1_id, sketch2_id):
        # prepare configuration
        config = Munch()

        # experiment realted
        config.org_sketch_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), f'ASIST_Dataset/Data/Data_A/{question_name}/{sketch1_id}.xml')
        config.tar_sketch_path = os.path.join(os.path.abspath(os.path.join(__file__ ,"../../..")), f'ASIST_Dataset/Data/Data_A/{question_name}/{sketch2_id}.xml')
        config.seed = 1
        config.verbose = 4 # 1-3 for normal messages, 4 to save visualizations and 5 to show visualization
        config.re_sampling = 0.0
        config.mx_dissimilarity = 50
        config.construction_step_size = 0.001
        config.mn_len = 5
        config.video_dir = '../../generated_videos/{0}/{1}'.format(question_name, sketch1_id)
        config.save_video_path = os.path.join(config.video_dir, f'{sketch2_id}.mp4')
        config.pretrained_model_path = ''
        config.fine_tune_epochs = 200

        config.org_flip = False
        config.org_shift_x = 0
        config.org_shift_y = 0
        config.tar_flip = False
        config.tar_shift_x = 0
        config.tar_shift_y = 0

        # saving and visualization related
        config.vis_video = False
        config.load_trans_params = False

        os.makedirs(config.video_dir, exist_ok=True)

        return config