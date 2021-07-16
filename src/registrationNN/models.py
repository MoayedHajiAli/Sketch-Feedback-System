import keras.backend as K
from keras.layers import Input, Dense, concatenate, Conv2D, Reshape, Flatten, LayerNormalization, MaxPool2D
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback
from keras.losses import mse
from keras.models import load_model
from keras.initializers import HeNormal
from keras import Model
from utils.ObjectUtil import ObjectUtil
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from utils.ObjectUtil import ObjectUtil
from utils.Utils import Utils
from utils.RegistrationUtils import RegistrationUtils
from animator.SketchAnimation import SketchAnimation
from sketchformer.builders.layers.transformer import Encoder, SelfAttnV1
import copy
from matplotlib import pyplot as plt
import random as rnd
import os
from tensorflow.python.client import device_lib
from datetime import datetime
import time

class registration_model:
    def knn_loss(self, sketches, p):
        # constants
        scaling_f = 5
        shearing_f = 5
        rotation_f = 1.5

        sketches = tf.identity(sketches)
        # sketches (batch, 2, 126, 3, 1)  p(batch, 7)
        org = sketches[:, 0, :, :2, 0]
        tar = sketches[:, 1, :, :2, 0]
        # org: (batch, 126, 2)  tar: (batch, 126, 2)
        org_pen = sketches[:, 0, :, 2, 0]
        tar_pen = sketches[:, 1, :, 2, 0]
        # org_pen: (batch, 126) tar_pen: (batch, 126)   represents the pen state in stroke-3 format

        # obtain transformation matrix parameters
        t = []
        t.append(p[:, 0] * (K.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * K.sin(p[:, 2])))
        t.append(p[:, 0] * (p[:, 3] * K.cos(p[:, 2]) - K.sin(p[:, 2])))
        t.append(p[:, 5])
        t.append(p[:, 1] * (K.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * K.cos(p[:, 2])))
        t.append(p[:, 1] * (p[:, 3] * K.sin(p[:, 2]) + K.cos(p[:, 2])))
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
        # add scaling cost
        # tran_cost = K.sum(tf.math.maximum(K.square(p[:, 0]), 1 / K.square(p[:, 0])) * scaling_f)
        # tran_cost += K.sum(tf.math.maximum(p[:, 1] ** 2, 1 / (p[:, 1] ** 2)) * scaling_f)
        # # add roation cost
        # tran_cost += K.sum((p[:, 2] ** 2) * rotation_f)
        # # add shearing cost
        # tran_cost += K.sum((p[:, 3] ** 2) * shearing_f)
        # tran_cost += K.sum((p[:, 4] ** 2) * shearing_f)
        # add shearing cost

        return sm_cost # + K.sqrt(tran_cost)         

    @staticmethod
    def np_knn_loss(sketches, p, maxlen=128):
        # constants
        scaling_f = 5
        shearing_f = 5
        rotation_f = 1.5
        
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
        t = []
        t.append(p[:, 0] * (np.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * np.sin(p[:, 2])))
        t.append(p[:, 0] * (p[:, 3] * np.cos(p[:, 2]) - np.sin(p[:, 2])))
        t.append(p[:, 5])
        t.append(p[:, 1] * (np.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * np.cos(p[:, 2])))
        t.append(p[:, 1] * (p[:, 3] * np.sin(p[:, 2]) + np.cos(p[:, 2])))
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

        # # add scaling cost
        # tran_cost = sum(max((p[:, 0] ** 2), 1 / (p[:, 0] ** 2)) * scaling_f)
        # tran_cost += sum(max(p[:, 1] ** 2, 1 / (p[:, 1] ** 2)) * scaling_f)
        # # add roation cost
        # tran_cost += sum((p[:, 2] ** 2) * rotation_f)
        # # add shearing cost
        # tran_cost += sum((p[:, 3] ** 2) * shearing_f)
        # tran_cost += sum((p[:, 4] ** 2) * shearing_f)
        # add shearing cos

        return sm_cost # + np.sqrt(tran_cost)      
    

    def init_model(self):
        # build the model with stroke-3 format
        org_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        org_reshaped = Reshape((128, 3))(org_inputs)
        tar_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        tar_reshaped = Reshape((128, 3))(tar_inputs)

        org_fe_layer0 = Dense(64, activation='relu')(org_reshaped)
        org_fe_layer0 = LayerNormalization()(org_fe_layer0)
        org_fe_layer0 = Reshape((128, 64, 1))(org_fe_layer0)
        org_fe_layer1 = Conv2D(1, (3, 3), 1)(org_fe_layer0) 
        org_fe_layer1 = MaxPool2D((3, 3))(org_fe_layer1)
        org_fe_layer1 = Reshape((42 * 20,)) (org_fe_layer1)
        # org_fe_layer1 = Flatten()(org_fe_layer0)

        # org_enc = Encoder(1, 128, 4, 256, None)(org_inputs)
        # org_embd = SelfAttnV1(128)(org_enc)

        org_fe_layer2= Dense(64, activation='relu')(org_fe_layer1)
        org_fe_layer2 = LayerNormalization()(org_fe_layer2)
        org_fe_layer3= Dense(32, activation='relu')(org_fe_layer2)
        org_fe_layer3 = LayerNormalization()(org_fe_layer3)
        org_fe = Model(org_inputs, org_fe_layer3)


        tar_fe_layer0= Dense(64, activation='relu')(tar_reshaped)
        tar_fe_layer0 = LayerNormalization()(tar_fe_layer0)
        tar_fe_layer0 = Reshape((128, 64, 1))(tar_fe_layer0)
        tar_fe_layer1 = Conv2D(1, (3, 3), 1)(tar_fe_layer0)
        tar_fe_layer1 = MaxPool2D((3, 3))(tar_fe_layer1)
        tar_fe_layer1 = Reshape((42 * 20, )) (tar_fe_layer1)
        # tar_fe_layer1 = Flatten()(tar_fe_layer0)

        # tar_enc = Encoder(1, 128, 4, 256, None)(tar_inputs)
        # tar_embd = SelfAttnV1(128)(tar_enc)
        tar_fe_layer2 = Dense(64, activation='relu')(tar_fe_layer1)
        tar_fe_layer2 = LayerNormalization()(tar_fe_layer2)
        tar_fe_layer3 = Dense(32, activation='relu')(tar_fe_layer2)
        tar_fe_layer3 = LayerNormalization()(tar_fe_layer3)
        tar_fe = Model(tar_inputs, tar_fe_layer3)

        merged_fe = concatenate([org_fe.output, tar_fe.output])
        # original and target extracted features
        layer1 = Dense(32, activation='relu', kernel_initializer=HeNormal(seed=5))(merged_fe)
        params = Dense(7, activation="linear", kernel_initializer=HeNormal(seed=6))(layer1)

        self.model = Model(inputs=[org_fe.input, tar_fe.input], outputs=params)
        self.model.compile(loss=self.knn_loss, optimizer=Adam(learning_rate=self.model_config.learning_rate))

    
    def fit(self, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches):
        print(train_org_sketches.shape, train_tar_sketches.shape, val_org_sketches.shape, val_tar_sketches.shape)
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
                    print("Start epoch {} of training; params predictions: {}".format(epoch, params[0:1]))
                    loss = registration_model.np_knn_loss(cmb_sketches[0:1], params[0:1])
                    print("Start epoch {} of training; loss: {}".format(epoch, loss))

                def on_epoch_end(self, epoch, logs={}):
                    params = self.model.predict((org_sketches, tar_sketches))
                    print("End epoch {} of training; params predictions: {}".format(epoch, params[0:1]))
                    loss = registration_model.np_knn_loss(cmb_sketches[0:1], params[0:1])
                    print("End epoch {} of training; loss: {}".format(epoch, loss))
            
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

        # save train loss curve
        ax = plt.subplot()
        ax.set_title("Training loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("loss")
        ax.plot(model_history.history['loss'])
        plt.savefig(self.model_config.vis_dir + '/train_loss.png')

        # save validation loss curve 
        ax = plt.subplot()
        ax.set_title("Testing loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("loss")
        ax.plot(model_history.history['val_loss'])
        plt.savefig(self.model_config.vis_dir + '/test_loss.png')
        

    def __init__(self,  model_config):
        super().__init__()

        self.model_config = model_config

        if model_config.load:
            print("[model.py] loading saved model of experiment", model_config.exp_id)
            self.model = load_model(model_config.exp_dir, custom_objects={'knn_loss': self.knn_loss})
        else:
            self.init_model()
            
    
    def predict(self, org_sketches, tar_sketches):
        return self.model.predict((org_sketches, tar_sketches))
        # tf.keras.utils.plot_model(self.model, to_file='model.png', show_layer_names=True, rankdir='TB', show_shapes=True)
        


class model_visualizer():
    def visualize_model(model, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, model_config):
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
        org_objs = ObjectUtil.accumalitive_stroke3_to_poly(org_sketches)
        tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(tar_sketches)
        val_org_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_org_sketches)
        val_tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_tar_sketches)

        if model_config.vis_transformation:
            # animate the tranformation
            print("[models.py] visualizing transformation")
            params = model.predict(org_sketches, tar_sketches)
            inds = rnd.choices(range(len(org_sketches)), k=model_config.num_vis_samples)

            for i in inds:
                org_objs[i].transform(RegistrationUtils.obtain_transformation_matrix(params[i]))
            
            for i in inds:
                animation = SketchAnimation([org_objs[i]], [tar_objs[i]]) 
                animation.seq_animate_all([params[i]]) # question: objects are already transformed. Do we need to reset?
                org_transformed[i].transform(RegistrationUtils.obtain_transformation_matrix(params[i])) # this part is not working. TODO: Fix

        if model_config.vis_train:
            # visualize samples of transformation without animation
            vis_dir = os.path.join(model_config.vis_dir, 'on_training')
            os.makedirs(vis_dir, exist_ok=True)

            print(f"[models.py] {time.ctime()}: Saving training visualizations")
            params = model.predict(org_sketches, tar_sketches)
            for j in range(model_config.num_vis_samples):
                inds = rnd.choices(range(len(org_objs)), k=5)
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

        # visualizing the validation
        if model_config.vis_test:
            # predict transformation
            params = model.predict(val_org_sketches, val_tar_sketches)
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
                    params = model.predict(val_org_sketches, val_tar_sketches)
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

                params = model.predict(val_org_sketches, val_tar_sketches)
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