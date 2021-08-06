import keras.backend as K
from keras.layers import Input, Dense, concatenate, Conv2D, Reshape, Flatten, LayerNormalization, MaxPool2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from keras.losses import mse
from keras.models import load_model
from keras.initializers import HeNormal
from keras import Model
from utils.ObjectUtil import ObjectUtil
from sketch_object.UnlabeledObject import UnlabeledObject
import numpy as np
import tensorflow as tf
from utils.ObjectUtil import ObjectUtil
from utils.Utils import Utils
from utils.RegistrationUtils import RegistrationUtils
from animator.SketchAnimation import SketchAnimation
from sketchformer.builders.layers.transformer import UnmaskedEncoder, SelfAttnV3
import copy
from matplotlib import pyplot as plt
import random as rnd
import os
from tensorflow.python.client import device_lib
from datetime import datetime
import time
from multiprocessing import Pool
from scipy.optimize import minimize, basinhopping, approx_fprime
import time

class NNModel:
    """For a given orginal and target object, find the transformation paramters 
    (7 parameters, with respect to the 0,0 origin) of the original object so that 
    after the transformation, it best aligns with the target object
    """

    def get_knn_loss(self, scaling_f, shearing_f, rotation_f):
        def knn_loss(sketches, p):

            sketches = tf.identity(sketches)
            # sketches (batch, 2, 126, 3, 1)  p(batch, 7)
            org = sketches[:, 0, :, :2, 0]
            tar = sketches[:, 1, :, :2, 0]
            # org: (batch, 126, 2)  tar: (batch, 126, 2)
            org_pen = sketches[:, 0, :, 2, 0]
            tar_pen = sketches[:, 1, :, 2, 0]
            # org_pen: (batch, 126) tar_pen: (batch, 126)   represents the pen state in stroke-3 format

            # obtain transformation matrix parameters
            # t = []
            # t.append(p[:, 0] * (K.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * K.sin(p[:, 2])))
            # t.append(p[:, 0] * (p[:, 3] * K.cos(p[:, 2]) - K.sin(p[:, 2])))
            # t.append(p[:, 5])
            # t.append(p[:, 1] * (K.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * K.cos(p[:, 2])))
            # t.append(p[:, 1] * (p[:, 3] * K.sin(p[:, 2]) + K.cos(p[:, 2])))
            # t.append(p[:, 6])

            t = []
            t.append(p[:, 0] * (K.cos(p[:, 2]) - p[:, 4] * K.sin(p[:, 2])))
            t.append(-1 * p[:, 1] * K.sin(p[:, 2]))
            t.append(p[:, 5])
            t.append(p[:, 0] * (K.sin(p[:, 2]) + p[:, 4] * K.cos(p[:, 2])))
            t.append(p[:, 1] * K.cos(p[:, 2]))
            t.append(p[:, 6])
            t = K.expand_dims(t, -1)
            # t: (batch, 6, 1)

            # apply transformation on all points in original 
            org_x = org[:, :, 0] * t[0] + org[:, :, 1] * t[1] + t[2]
            org_y = org[:, :, 0] * t[3] + org[:, :, 1] * t[4] + t[5]

            # org_x: represent x coords (batch, 126)
            # org_y: represent x coords (batch, 126)
            org_x = K.expand_dims(org_x, 2)
            org_y = K.expand_dims(org_y, 2)
            # org_x: (represent x coords) (batch, 126, 1)
            # org_y: (represent x coords) (batch, 126, 1)

            org_cmb = tf.concat([org_x, org_y], axis=-1)
            # org_cmb : (batch, 126, 2)

            org_cmb = K.expand_dims(org_cmb, 1)
            # org_cmb: (batch, 1, 126, 2)

            tar_cmb = K.expand_dims(tar, 2)
            # tar_cmb: (batch, 126, 1, 2)

            # obtain pairwise differences between original and target sketches
            diff = org_cmb - tar_cmb
            # diff: (batch, 126, 126, 2)
    
            sm = K.sum(diff ** 2, axis=-1)
            sm_sqrt = K.sqrt(sm)
            # sm_sqrt: (batch, 126, 126)

            # obtain nearest points from org->tar + from tar->org
            mn = K.min(sm_sqrt, axis=-2) * (1 - org_pen) + K.min(sm_sqrt, axis=-1) * (1 - tar_pen)
            # mn: (batch, 126)

            sm_cost = K.sum(mn, axis=1) 
            # sm_cost: (batch, )

            # normalize with the number of points
            sm_cost /= 128 - K.sum(org_pen, axis=-1) + 128 - K.sum(tar_pen, axis=-1) 

            # add penalty to the transformation parameters
            # @size p: (batch, 7)
            # # add scaling cost
            tran_cost = K.sum(tf.math.maximum(K.square(p[:, 0]), 1 / K.square(p[:, 0])) * scaling_f)
            tran_cost += K.sum(tf.math.maximum(p[:, 1] ** 2, 1 / (p[:, 1] ** 2)) * scaling_f)
            # add roation cost
            tran_cost += K.sum((p[:, 2] ** 2) * rotation_f)
            # add shearing cost
            tran_cost += K.sum((p[:, 3] ** 2) * shearing_f)
            tran_cost += K.sum((p[:, 4] ** 2) * shearing_f)
            # add shearing cost

            return sm_cost  + K.sqrt(tran_cost)    
        return knn_loss     

    @staticmethod
    def np_knn_loss(sketches, p, maxlen=128):
        
        sketches = copy.deepcopy(sketches)
        # sketches = tf.identity(sketches)
        # sketches (batch, 2, 126, 3, 1)  p(batch, 7)
        org = sketches[:, 0, :, :2, 0]
        tar = sketches[:, 1, :, :2, 0]
        # org: (batch, 126, 2)  tar: (batch, 126, 2)
        org_pen = sketches[:, 0, :, 2, 0]
        tar_pen = sketches[:, 1, :, 2, 0]
        # org_pen: (batch, 126) tar_pen: (batch, 126)   represents the pen state in stroke-3 format

        # obtain transformation matrix parameters
        # t = []
        # t.append(p[:, 0] * (np.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * np.sin(p[:, 2])))
        # t.append(p[:, 0] * (p[:, 3] * np.cos(p[:, 2]) - np.sin(p[:, 2])))
        # t.append(p[:, 5])
        # t.append(p[:, 1] * (np.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * np.cos(p[:, 2])))
        # t.append(p[:, 1] * (p[:, 3] * np.sin(p[:, 2]) + np.cos(p[:, 2])))
        # t.append(p[:, 6])

        t = []
        t.append(p[:, 0] * (np.cos(p[:, 2]) - p[:, 4] * np.sin(p[:, 2])))
        t.append(-1 * p[:, 1] * np.sin(p[:, 2]))
        t.append(p[:, 5])
        t.append(p[:, 0] * (np.sin(p[:, 2]) + p[:, 4] * np.cos(p[:, 2])))
        t.append(p[:, 1] * np.cos(p[:, 2]))
        t.append(p[:, 6])
        t = np.expand_dims(t, -1)
        # t: (batch, 6, 1)

        # apply transformation on all points in original 
        org_x = org[:, :, 0] * t[0] + org[:, :, 1] * t[1] + t[2]
        org_y = org[:, :, 0] * t[3] + org[:, :, 1] * t[4] + t[5]

        # org_x: represent x coords (batch, 126)
        # org_y: represent x coords (batch, 126)
        org_x = np.expand_dims(org_x, 2)
        org_y = np.expand_dims(org_y, 2)
        # org_x: (represent x coords) (batch, 126, 1)
        # org_y: (represent x coords) (batch, 126, 1)

        org_cmb = np.concatenate([org_x, org_y], axis=-1)
        # org_cmb : (batch, 126, 2)
        org_cmb = np.expand_dims(org_cmb, 1)
        # org_cmb: (batch, 1, 126, 2)

        tar_cmb = np.expand_dims(tar, 2)
        # tar_cmb: (batch, 126, 1, 2)

        # obtain pairwise differences between original and target sketches
        diff = org_cmb - tar_cmb
        # diff: (batch, 126, 126, 2)

        sm = np.sum(diff ** 2, axis=-1)
        sm_sqrt = np.sqrt(sm)
        # sm_sqrt: (batch, 126, 126)

        # obtain nearest points from org->tar + from tar->org
        mn = np.min(sm_sqrt, axis=-2) * (1 - org_pen) + np.min(sm_sqrt, axis=-1) * (1 - tar_pen)
        # mn: (batch, 126)

        sm_cost = np.sum(mn, axis=1) 
        # sm_cost: (batch, )

        # normalize with the number of points
        sm_cost /= maxlen - np.sum(org_pen, axis=-1) + maxlen - np.sum(tar_pen, axis=-1) 

        # # # add scaling cost
        # tran_cost = np.sum(np.maximum((p[:, 0] ** 2), 1 / (p[:, 0] ** 2)) * scaling_f)
        # tran_cost += np.sum(np.maximum(p[:, 1] ** 2, 1 / (p[:, 1] ** 2)) * scaling_f)
        # # add roation cost
        # tran_cost += np.sum((p[:, 2] ** 2) * rotation_f)
        # # add shearing cost
        # tran_cost += np.sum((p[:, 3] ** 2) * shearing_f)
        # tran_cost += np.sum((p[:, 4] ** 2) * shearing_f)
        # # add shearing cos

        return sm_cost # + np.sqrt(tran_cost)  
    

    def init_model(self):
        # build the model with stroke-3 format
        org_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        org_reshaped = Reshape((128, 3))(org_inputs)
        tar_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        tar_reshaped = Reshape((128, 3))(tar_inputs)
        
        # Conv encoder
        org_fe_layer0 = Dense(64, activation='relu')(org_reshaped) # 128, 64
        org_fe_layer0 = LayerNormalization()(org_fe_layer0)
        org_fe_layer0 = Reshape((128, 64, 1))(org_fe_layer0)
        org_fe_layer1 = Conv2D(1, (3, 3), 1)(org_fe_layer0) 
        org_fe_layer1 = MaxPool2D((3, 3))(org_fe_layer1)
        org_fe_layer1 = Reshape((42 * 20,)) (org_fe_layer1)

        # sketchformer encoder + self attention
        # org_enc = UnmaskedEncoder(
        #     num_layers=1,
        #     d_model=128,
        #     num_heads=4, dff=256,
        #     input_vocab_size=None, rate=0.1,
        #     use_continuous_input=True)(org_reshaped)

        # org_fe_layer1 = SelfAttnV3(128)(org_enc)    
        
        
        org_fe_layer2= Dense(64, activation='relu')(org_fe_layer1)
        org_fe_layer2 = LayerNormalization()(org_fe_layer2)
        org_fe_layer3= Dense(32, activation='relu')(org_fe_layer2)
        org_fe_layer3 = LayerNormalization()(org_fe_layer3)
        org_fe = Model(org_inputs, org_fe_layer3)
        

        # Conv encoder 
        tar_fe_layer0= Dense(64, activation='relu')(tar_reshaped)
        tar_fe_layer0 = LayerNormalization()(tar_fe_layer0)
        tar_fe_layer0 = Reshape((128, 64, 1))(tar_fe_layer0)
        tar_fe_layer1 = Conv2D(1, (3, 3), 1)(tar_fe_layer0)
        tar_fe_layer1 = MaxPool2D((3, 3))(tar_fe_layer1)
        tar_fe_layer1 = Reshape((42 * 20, )) (tar_fe_layer1)

        # sketchformer encoder + self attention

        # tar_enc = UnmaskedEncoder(
        #     num_layers=2,
        #     d_model=128,
        #     num_heads=4, dff=256,
        #     input_vocab_size=None, rate=0.1,
        #     use_continuous_input=True)(tar_reshaped)
        # tar_fe_layer1 = SelfAttnV3(128)(tar_enc)


        tar_fe_layer2 = Dense(64, activation='relu')(tar_fe_layer1)
        tar_fe_layer2 = LayerNormalization()(tar_fe_layer2)
        tar_fe_layer3 = Dense(32, activation='relu')(tar_fe_layer2)
        tar_fe_layer3 = LayerNormalization()(tar_fe_layer3) # 32
        tar_fe = Model(tar_inputs, tar_fe_layer3)

        merged_fe = concatenate([org_fe.output, tar_fe.output])
        # original and target extracted features
        layer1 = Dense(32, activation='relu', kernel_initializer=HeNormal(seed=5))(merged_fe)
        params = Dense(7, activation="linear", kernel_initializer=HeNormal(seed=6))(layer1)

        self.model = Model(inputs=[org_fe.input, tar_fe.input], outputs=params)
        self.model.compile(loss=self.get_knn_loss(self.model_config.scaling_f, self.model_config.shearing_f, self.model_config.rotation_f),
                          optimizer=Adam(learning_rate=self.model_config.learning_rate))

    
    def fine_tune(self, train_org_sketches, train_tar_sketches, epochs):
        # convert sketches into stroke-3 format
        train_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_org_sketches)
        train_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_tar_sketches)
        
        # add padding
        train_org_sketches = RegistrationUtils.pad_sketches(train_org_sketches, maxlen=128)
        train_tar_sketches = RegistrationUtils.pad_sketches(train_tar_sketches, maxlen=128)
        org_sketches, tar_sketches = train_org_sketches, train_tar_sketches

        org_sketches = np.expand_dims(org_sketches, axis=-1)
        tar_sketches = np.expand_dims(tar_sketches, axis=-1)
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)

        # fit model
        model_history = self.model.fit(x=[org_sketches, tar_sketches], 
                                       y=cmb_sketches, 
                                       batch_size=self.model_config.batch_size,
                                       epochs=epochs)
        return model_history.history
            
        


    def fit(self, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches):
        print("Devices:", device_lib.list_local_devices())
        print(f"length of original sketchs:{len(train_org_sketches)}")
        # convert sketches into stroke-3 format
        train_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_org_sketches)
        train_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_tar_sketches)
        val_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_org_sketches)
        val_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_tar_sketches)
        
        # add padding
        train_org_sketches = RegistrationUtils.pad_sketches(train_org_sketches, maxlen=128)
        train_tar_sketches = RegistrationUtils.pad_sketches(train_tar_sketches, maxlen=128)
        val_org_sketches = RegistrationUtils.pad_sketches(val_org_sketches, maxlen=128)
        val_tar_sketches = RegistrationUtils.pad_sketches(val_tar_sketches, maxlen=128)
        org_sketches, tar_sketches = train_org_sketches, train_tar_sketches

        print(f"[models.py] {time.ctime()}: finshed loading the data")

        org_sketches = np.expand_dims(org_sketches, axis=-1)
        tar_sketches = np.expand_dims(tar_sketches, axis=-1)
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)

        # validation
        val_org_sketches = np.expand_dims(val_org_sketches, axis=-1)
        val_tar_sketches = np.expand_dims(val_tar_sketches, axis=-1)
        val_cmb_sketches = np.stack((val_org_sketches, val_tar_sketches), axis=1)

        # prepare call backs
        class epoch_callback(Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    params = self.model.predict((org_sketches, tar_sketches))
                    print("Start epoch {} of training; params predictions of first sketch: {}".format(epoch, params[0:1]))
                    loss = NNModel.np_knn_loss(cmb_sketches[0:1], params[0:1])
                    print("Start epoch {} of training; loss of first sketch: {}".format(epoch, loss))
            
        if self.model_config.load_ckpt:
            # restore latest checkpoint
            latest_ckpt = tf.train.latest_checkpoint(self.model_config.exp_dir)
            print("[model.py] latest checkpoint is:", latest_ckpt)
            self.model.load_weights(latest_ckpt)
        
        clbks = []
        if self.model_config.save_ckpt:
            clbks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.model_config.ckpt_path,
                                                save_weights_only=True,
                                                monitor='loss',
                                                mode='min',
                                                save_best_only=self.model_config.save_best_only,
                                                verbose=10))
        if self.model_config.verbose > 3:
            clbks.append(epoch_callback())

        # fit model
        model_history = self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=self.model_config.batch_size, \
                            epochs=self.model_config.epochs, callbacks=clbks, validation_data=([val_org_sketches, val_tar_sketches], val_cmb_sketches))
            
        
        # save the whole model 
        if self.model_config.save:
            print("[model.py] saving new model of experiment", self.model_config.exp_id)
            self.model.save(self.model_config.exp_dir)
        
        # save the model config files
        Utils.save_obj_pkl(model_history.history, self.model_config.hist_path)

        if 'loss' in model_history.history.keys():
            # save train loss curve
            fig, ax = plt.subplots()
            ax.set_title("Training loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            ax.plot(model_history.history['loss'])
            fig.savefig(self.model_config.vis_dir + '/train_loss.png')

        if 'val_loss' in model_history.history.keys():
            # save validation loss curve 
            fig, ax = plt.subplots()
            ax.set_title("Testing loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            ax.plot(model_history.history['val_loss'])
            fig.savefig(self.model_config.vis_dir + '/test_loss.png')
        

    def __init__(self,  model_config):
        super().__init__()

        self.model_config = model_config

        if model_config.load:
            print("[model.py] loading saved model of experiment", model_config.exp_id)
            self.model = load_model(model_config.exp_dir,
                         custom_objects={'knn_loss': self.get_knn_loss(self.model_config.scaling_f, self.model_config.shearing_f, self.model_config.rotation_f)})
        else:
            self.init_model()
            
    
    def predict(self, org_obj, tar_obj):
        """ind the transformation alignment parameters for each of the org_sketches to its correspondance in the tar_sketches

        Args:
            org_obj (list(UnlabeldObject)): list of N sketches
            tar_obj (list(UnlabeldObject)): list of target N sketches
        """
        # convert sketches into stroke-3 format
        org_obj_stroke3 = ObjectUtil.poly_to_accumulative_stroke3(org_obj)
        tar_obj_stroke3 = ObjectUtil.poly_to_accumulative_stroke3(tar_obj)
        
        # add padding
        org_obj_stroke3 = RegistrationUtils.pad_sketches(org_obj_stroke3, maxlen=128)
        tar_obj_stroke3 = RegistrationUtils.pad_sketches(tar_obj_stroke3, maxlen=128)

        org_obj_stroke3 = np.expand_dims(org_obj_stroke3, axis=-1)
        tar_obj_stroke3 = np.expand_dims(tar_obj_stroke3, axis=-1)
        cmb_obj_stroke3 = np.stack((org_obj_stroke3, tar_obj_stroke3), axis=1)

        params = self.predict_from_stroke3(org_obj_stroke3, tar_obj_stroke3)

        # find the loss
        losses = self.np_knn_loss(cmb_obj_stroke3, params)
        return params, losses
 

    def predict_from_stroke3(self, org_obj, tar_obj):
        """ find the transformation alignment parameters for each of the org_sketches to its correspondance in the tar_sketches

        Args:
            org_obj (array(B x 128 x 3 x 1)): list of B sketches to aligne in stroke-3 format
            tar_obj (array(B x 128 x 3 x 1)): list of B sketches to be aligned to in stroke-3 format

        Returns:
            [array] : B x 7 list of transformation parameters in the corresponding order of : 
        """
        return self.model.predict((org_obj, tar_obj))
        # tf.keras.utils.plot_model(self.model, to_file='model.png', show_layer_names=True, rankdir='TB', show_shapes=True)
        


class model_visualizer():
    def visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config):
        
        org_trn_obj, tar_trn_obj = train_org_sketches, train_tar_sketches
        # convert sketches into stroke-3 format
        train_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_org_sketches)
        train_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_tar_sketches)
        val_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_org_sketches)
        val_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_tar_sketches)
        

        # add padding
        train_org_sketches = RegistrationUtils.pad_sketches(train_org_sketches, maxlen=128)
        train_tar_sketches = RegistrationUtils.pad_sketches(train_tar_sketches, maxlen=128)
        val_org_sketches = RegistrationUtils.pad_sketches(val_org_sketches, maxlen=128)
        val_tar_sketches = RegistrationUtils.pad_sketches(val_tar_sketches, maxlen=128)

        org_sketches, tar_sketches = train_org_sketches, train_tar_sketches
        org_sketches = np.expand_dims(org_sketches, axis=-1)
        tar_sketches = np.expand_dims(tar_sketches, axis=-1)
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)

        # validation
        val_org_sketches = np.expand_dims(val_org_sketches, axis=-1)
        val_tar_sketches = np.expand_dims(val_tar_sketches, axis=-1)
        val_cmb_sketches = np.stack((val_org_sketches, val_tar_sketches), axis=1)

        # obtain the poly represenation of the sketches to be used later in visualization and transformation
        # these objects will have the same normalization as the ones in used with the traning
        org_objs = ObjectUtil.accumalitive_stroke3_to_poly(org_sketches)
        tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(tar_sketches)
        val_org_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_org_sketches)
        val_tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_tar_sketches)

        inds = rnd.choices(range(len(org_sketches)), k=model_config.num_vis_samples)
        if model_config.vis_transformation:
            # animate the tranformation and save the videos
            vis_dir = os.path.join(model_config.vis_dir, 'transformation videos')
            os.makedirs(vis_dir, exist_ok=True)

            print("[models.py] visualizing transformation")
            params = model.predict_from_stroke3(org_sketches, tar_sketches)
            # inds = rnd.choices(range(len(org_sketches)), k=model_config.num_vis_samples)

            # for i in inds: TODO: no need for such step. Delete
            #     # transform objects in order to update the step vector
            #     org_objs[i].transform(RegistrationUtils.obtain_transformation_matrix(params[i]))
            
            for i in inds:
                animation = SketchAnimation([org_trn_obj[i]], [tar_trn_obj[i]]) 
                animation.seq_animate_all([params[i]], 
                                         denormalize_trans=True,
                                         save=model_config.save_transformation_vis, 
                                         file=os.path.join(vis_dir, f'example_{i}.mp4')) 
                org_trn_obj[i].reset()
                tar_trn_obj[i].reset()


        if model_config.vis_train:
            # visualize samples of transformation of original objects with the denormalized matrix
            vis_dir = os.path.join(model_config.vis_dir, 'training_denormalized')
            os.makedirs(vis_dir, exist_ok=True)

            print(f"[models.py] {time.ctime()}: Saving training visualizations")
            params = model.predict_from_stroke3(org_sketches, tar_sketches)
            for j in range(model_config.num_vis_samples):
                inds = rnd.choices(range(len(org_objs)), k=5)
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    org_trn_obj[ind].reset()
                    org_trn_obj[ind].visualize(ax=axs[i][0], show=False)
                    tar_trn_obj[ind].visualize(ax=axs[i][1], show=False)
                    tar_trn_obj[ind].visualize(ax=axs[i][2], show=False)
                    org_trn_obj[ind].transform(ObjectUtil.denormalized_transformation(
                        org_trn_obj[ind],
                        tar_trn_obj[ind],
                        RegistrationUtils.obtain_transformation_matrix(params[ind])), object_min_origin=True)
                    org_trn_obj[ind].visualize(ax=axs[i][2], show=False)
                    org_trn_obj[ind].reset()
                    
                plt.savefig(vis_dir + "/res{0}.png".format(j))

        if model_config.vis_train:
            # visualize samples of transformation without animation
            vis_dir = os.path.join(model_config.vis_dir, 'training')
            os.makedirs(vis_dir, exist_ok=True)

            print(f"[models.py] {time.ctime()}: Saving training visualizations")
            params = model.predict_from_stroke3(org_sketches, tar_sketches)
            for j in range(model_config.num_vis_samples):
                inds = rnd.choices(range(len(org_objs)), k=5)
                print(inds)
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    org_objs[ind].reset()
                    org_objs[ind].visualize(ax=axs[i][0], show=False)
                    tar_objs[ind].visualize(ax=axs[i][1], show=False)
                    tar_objs[ind].visualize(ax=axs[i][2], show=False)
                    org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                    org_objs[ind].visualize(ax=axs[i][2], show=False)
                    org_objs[ind].reset()
                    
                plt.savefig(vis_dir + "/res{0}.png".format(j))

        if model_config.vis_train:
            # visualize samples of transformation of original object, by obtaining sequential transformation
            vis_dir = os.path.join(model_config.vis_dir, 'training-sequential')
            os.makedirs(vis_dir, exist_ok=True)

            print(f"[models.py] {time.ctime()}: Saving training visualizations")
            params = model.predict_from_stroke3(org_sketches, tar_sketches)
            for j in range(model_config.num_vis_samples):
                inds = rnd.choices(range(len(org_trn_obj)), k=5)
                print(inds)
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    org_trn_obj[ind].reset()
                    org_trn_obj[ind].visualize(ax=axs[i][0], show=False)
                    tar_trn_obj[ind].visualize(ax=axs[i][1], show=False)
                    tar_trn_obj[ind].visualize(ax=axs[i][2], show=False)
                    params_6 = RegistrationUtils.obtain_transformation_matrix(params[ind])
                    # denormalize
                    params_6 = ObjectUtil.denormalized_transformation(
                        org_trn_obj[ind],
                        tar_trn_obj[ind],
                        params_6)
                    
                    params_7 = RegistrationUtils.decompose_tranformation_matrix(params_6)
                    # get sequential
                    seq_params = RegistrationUtils.get_seq_translation_matrices(params_7)
                    for k, tmp in enumerate(seq_params):
                        if k == len(seq_params) - 1:
                            org_trn_obj[ind].transform(tmp, object_min_origin=True, retain_origin=False)
                        else:    
                            org_trn_obj[ind].transform(tmp, object_min_origin=True, retain_origin=True)
                    org_trn_obj[ind].visualize(ax=axs[i][2], show=False)
                    org_trn_obj[ind].reset()
                    
                plt.savefig(vis_dir + "/res{0}.png".format(j))
        # visualizing the validation
        if model_config.vis_test:
            # predict transformation
            params = model.predict_from_stroke3(val_org_sketches, val_tar_sketches)
            print("[model.py] resulted average loss without refinment", np.mean(model.np_knn_loss(val_cmb_sketches, params)))

            vis_dir = os.path.join(model_config.vis_dir, 'testing', 'wo_refinment')
            os.makedirs(vis_dir, exist_ok=True)

            print(f"[models.py] {time.ctime()}: Saving testing visualizations")
            # fix indices to comapre different approaches
            vis_inds = [np.concatenate([[0], rnd.choices(range(len(val_org_sketches)), k=4)]) for _ in range(model_config.num_vis_samples)]
            
            for j in range(model_config.num_vis_samples):
                inds = vis_inds[j]    
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    val_org_objs[ind].reset()
                    val_org_objs[ind].visualize(ax=axs[i][0], show=False)
                    val_tar_objs[ind].visualize(ax=axs[i][1], show=False)
                    val_tar_objs[ind].visualize(ax=axs[i][2], show=False)
                    val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                    val_org_objs[ind].visualize(ax=axs[i][2], show=False)
                    val_org_objs[ind].reset()

                plt.savefig(vis_dir + "/res{0}.png".format(j))
            
            if model_config.iter_refine_prediction:
                print(f"[model.py] {time.ctime()}: predicting with iterative refinement")
                vis_dir = os.path.join(model_config.vis_dir, 'testing', 'w_refinment')
                os.makedirs(vis_dir, exist_ok=True)

                for i in range(model_config.num_vis_samples):
                    # predict tranformation in every iteration 
                    params = model.predict_from_stroke3(val_org_sketches, val_tar_sketches)
                    print("resulted average loss with iterative refinment at iter {0}: {1}".format(i, np.mean(model.np_knn_loss(val_cmb_sketches, params))))
                    inds = vis_inds[i]
                    fig, axs = plt.subplots(len(inds), 3)
                    for j, ind in enumerate(inds):
                        val_org_objs[ind].visualize(ax=axs[j][0], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][1], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][2], show=False)
                        val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                        val_org_objs[ind].visualize(ax=axs[j][2], show=False)
                    
                    plt.savefig(vis_dir + "/iter{0}.png".format(i))

                    # transform all validation sketches
                    for j in range(len(val_org_objs)):
                        if j in inds:
                            continue
                        val_org_objs[j].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))


                    # obtain stroke 3 representation
                    val_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_org_objs, red_rdp=False, normalize=False)
                    val_org_sketches = RegistrationUtils.pad_sketches(val_org_sketches, maxlen=128)
                    val_org_sketches = np.expand_dims(val_org_sketches, axis=-1)
                    val_cmb_sketches = np.stack((val_org_sketches, val_tar_sketches), axis=1)

                    # obtain objs from stroke 3 represtentation
                    val_org_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_org_sketches)

            if model_config.fine_tune:
                # TODO: fix as it is not working
                vis_dir = os.path.join(model_config.vis_dir, 'testing', 'fine_tune')
                os.makedirs(vis_dir, exist_ok=True)
                refine_history = model.model.fit(x=[val_org_sketches, val_tar_sketches], y=val_cmb_sketches, batch_size=model_config.batch_size, epochs=model_config.fine_tune_epochs)
                
                # visualize loss
                ax = plt.subplot()
                ax.set_title("Testing loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("loss")
                ax.plot(refine_history.history['loss'])
                plt.savefig(vis_dir + "/loss.png")

                params = model.predict_from_stroke3(val_org_sketches, val_tar_sketches)
                print(f"[models.py] {time.ctime()}: resulted average loss after refinment", np.mean(model.np_knn_loss(val_cmb_sketches, params)))
                print(f"[models.py] {time.ctime()}: Saving testing with fine-tune visualizations")

                for i in range(model_config.num_vis_samples):
                    inds = vis_inds[i]
                    fig, axs = plt.subplots(len(inds), 3)
                    for j, ind in enumerate(inds):
                        val_org_objs[ind].visualize(ax=axs[j][0], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][1], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][2], show=False)
                        val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                        val_org_objs[ind].visualize(ax=axs[j][2], show=False)
                        val_org_objs[ind].reset()

                    plt.savefig(vis_dir + "/res{0}.png".format(i))

# TODO: the follwoing two models were not tested after restructing the code
class RegisterTwoObjects:
    
    def __init__(self, ref_obj:UnlabeledObject, tar_obj:UnlabeledObject, cost_fun):
        self.tar_obj = tar_obj
        self.ref_obj = ref_obj
        self.total_cost = cost_fun

    ##################
    # TODO: delete this method
    # test numerical gradient optimization
    def num_optimize(self, t):
        eps = 0.001
        lr = 0.1
        t = np.array(t)
        cur, prev = 100, 0
        for _ in range(200): 
            if _ % 50 == 0:
                self.ref_obj.transform(RegistrationUtils.obtain_transformation_matrix(t))
                self.ref_obj.visualize()
                self.ref_obj.reset()
            # obtain the gradient
            grad = approx_fprime(t, RegistrationUtils.embedding_dissimilarity, eps, self.ref_obj, self.tar_obj)
            t -= lr * np.array(grad)

        return -1, t

    # total dissimilarity including the cost of the transformation
    def total_dissimalirity(self, p, params = True, target_dis=True, original_dis=True):
        tran_cost = self.total_cost(p, self.mn_x, self.mx_x, self.mn_y, self.mx_y, len(self.ref_obj))
        if params:
            p = RegistrationUtils.obtain_transformation_matrix(p)
        
        dissimilarity = RegistrationUtils.calc_dissimilarity(self.ref_obj, self.tar_obj, p, target_nn = self.target_nn, 
                                                            target_dis=target_dis, original_dis=original_dis) 
        return dissimilarity + (tran_cost / (len(self.ref_obj) + len(self.tar_obj)))   


    def optimize(self, p = None, params = True, target_dis=True, original_dis=True):
        """optimize the disimilarity function.
    
            Params: 
                p: the transoformation parameters 
                params: if True, the function expects and return an array of parameters of shape(7), which specify the tarnsformation
                        paramerts for scaling_x, scaling_y, rotations, shearing_x, shearing_y, translation_x, translation_y.
                        if False, the function expects and return an array of parameters of shape(6), which specify the tarnsformation
                        array values
            """ 
        # find t if not specifies
        if p is None:
            x_dif = self.tar_obj.origin_x - self.ref_obj.origin_x
            y_dif = self.tar_obj.origin_y - self.ref_obj.origin_y
            if params:
                # p = np.array([random.uniform(1, 2), random.uniform(1, 2), 0.0, random.uniform(0, 1), random.uniform(0, 1), x_dif, y_dif])
                p = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif])
            else:
                p = np.array([1.0, 0.0, x_dif, 0.0, 1.0, y_dif])  

        # track function for scipy minimize
        def _track(xk):
            print(xk)

        #self.target_nn = NearestSearch(self.tar_obj.get_x(), self.tar_obj.get_y())
        self.target_nn = None

        # calculate min/max coordinates for the referenced object
        self.mn_x, self.mx_x = min(self.ref_obj.get_x()), max(self.ref_obj.get_x())
        self.mn_y, self.mx_y = min(self.ref_obj.get_y()), max(self.ref_obj.get_y())

        minimizer_kwargs = {"method": "BFGS", "args" : (params, target_dis, original_dis)}
        res = basinhopping(self.total_dissimalirity, p, minimizer_kwargs=minimizer_kwargs, disp=True, niter=2)
        d, p = res.fun, res.x 
        return d, p


class BlackBoxModel:

    def __init__(self):
        self.sh_cost, self.tr_cost = RegistrationUtils._shearing_cost, RegistrationUtils._translation_cost, 
        self.ro_cost, self.sc_cost = RegistrationUtils._rotation_cost, RegistrationUtils._scaling_cost

    def predict(self, original_obj, target_obj):
        n, m = len(self.original_obj), len(self.target_obj)
        dim = max(n,m)
        self.res_matrix = np.zeros((dim, dim))
        self.tra_matrix = np.zeros((dim, dim, 7))   

        # prepare queue of regiteration objects for multiprocessing
        pro_queue = []
        for obj1 in self.original_obj:
            for obj2 in self.target_obj:
                pro_queue.append(RegisterTwoObjects(obj1, obj2, self.total_cost))

        # register all the objects using pooling
        res = []
        with Pool(self.core_cnt) as p:
            res = list(p.map(self._optimize, pro_queue))

        # fill the result in the res_matrix
        t = 0
        for i in range(dim):
            # t = np.random.rand(7)
            for j in range(dim):
                if i >= n or j >= m:
                    d, p = RegistrationUtils.inf, np.zeros(7)
                else:
                    d, p = res[t]
                    t += 1
                self.res_matrix[i, j] = d
                self.tra_matrix[i, j] = p
        self.res_matrix = np.asarray(self.res_matrix)

        return self.res_matrix, self.tra_matrix

    
    # wrapper function for calling optimize on a RegisterTwoObjects
    def _optimize(self, reg):
        x_dif = reg.tar_obj.origin_x - reg.ref_obj.origin_x
        y_dif = reg.tar_obj.origin_y - reg.ref_obj.origin_y
        p = np.array([1.0, 1.0, 0.0, 0.0, 0.0, x_dif, y_dif])
        return reg.optimize(p = p, params=True)


    # obtain total transformation **parameters** cost
    def total_cost(self, p, mn_x, mx_x, mn_y, mx_y, ln):
        tot = 0.0
        tot += self.sc_cost(p[0], p[1], ln)
        tot += self.ro_cost(p[2], ln)
        tot += self.sh_cost(p[3], p[4], mn_x, mn_y, mx_x, mx_y, ln)
        tot += self.tr_cost(p[5], p[6], ln)
        return tot


if __name__ == "__main__":
    pass