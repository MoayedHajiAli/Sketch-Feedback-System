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
from utils.RegistrationUtils import RegistrationUtils
from animator.SketchAnimation import SketchAnimation
from sketchformer.builders.layers.transformer import Encoder, SelfAttnV1
import copy
from matplotlib import pyplot as plt
import random as rnd
import os
from tensorflow.python.client import device_lib
from datetime import datetime

class registration_model:
  
    def knn_loss(self, sketches, p):
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

        return sm_cost         

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

        return sm_cost     


    def _pad_sketches(self, sketches, maxlen=128, inf=1e9):
        converted_sketches = []
        for i in range(len(sketches)):
            tmp = []
            if len(sketches[i]) >= maxlen:
                tmp = np.array(sketches[i][:maxlen-1])
            else:
                tmp = sketches[i]
            # add at least one padding
            extra = np.repeat(np.array([[inf, inf, 1]]), maxlen-len(tmp), axis=0)
            converted_sketches.append(np.concatenate((tmp, extra), axis=0))
        return np.asarray(converted_sketches)
    

    def init_model(self):
        # build the model with stroke-3 format
        org_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        org_reshaped = Reshape((128, 3))(org_inputs)
        tar_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        tar_reshaped = Reshape((128, 3))(tar_inputs)

        org_fe_layer0= Dense(64, activation='relu')(org_reshaped)
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
        self.model.compile(loss=self.knn_loss, optimizer=Adam(learning_rate=0.005), metrics=['accuracy'])

    def __init__(self, train_org_sketches, train_tar_sketches, val_org_sketches, val_tar_sketches, k=-1):
        print(device_lib.list_local_devices())
        print(len(train_org_sketches), len(train_tar_sketches))
        # convert sketches into stroke-3 format
        train_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_org_sketches)
        train_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_tar_sketches)
        val_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_org_sketches)
        val_tar_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_tar_sketches)

        # add padding
        train_org_sketches = self._pad_sketches(train_org_sketches, maxlen=128)
        train_tar_sketches = self._pad_sketches(train_tar_sketches, maxlen=128)
        val_org_sketches = self._pad_sketches(val_org_sketches, maxlen=128)
        val_tar_sketches = self._pad_sketches(val_tar_sketches, maxlen=128)

        print(len(train_org_sketches), len(train_tar_sketches))
        # add every pair of objects to the training set
        # org_sketches, tar_sketches = [], []
        # for i in range(len(train_sketches)):
        #     if k == -1:
        #         inds = range(len(train_sketches))
        #     else:
        #         inds = rnd.choices(range(len(train_sketches)), k=k)
        #     for j in inds:
        #         org_sketches.append(np.array(train_sketches[i]))
        #         tar_sketches.append(np.array(train_sketches[j]))
        org_sketches, tar_sketches = train_org_sketches, train_tar_sketches
        # add every pair of objects to the training set
        # val_org_sketches, val_tar_sketches = [], []
        # for i in range(len(val_sketches)):
        #     for j in range(len(val_sketches)):
        #         val_org_sketches.append(np.array(val_sketches[i]))
        #         val_tar_sketches.append(np.array(val_sketches[j]))
        
        print("[models.py] finshed loading the data")
        org_sketches = np.expand_dims(org_sketches, axis=-1)
        tar_sketches = np.expand_dims(tar_sketches, axis=-1)
        print(len(org_sketches))
        print(len(tar_sketches))
        print(org_sketches.shape)
        print(tar_sketches.shape)
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)

        # validation
        val_org_sketches = np.expand_dims(val_org_sketches, axis=-1)
        val_tar_sketches = np.expand_dims(val_tar_sketches, axis=-1)
        val_cmb_sketches = np.stack((val_org_sketches, val_tar_sketches), axis=1)

        experiment_id = 5
        batch_size = 256
        num_epochs = 10
        load = False
        load_cp = False
        save = True
        vis_train = True
        vis_test = True
        refine_prediction = True
        iter_refine_prediction = True
        cp_dir = "../registrationNN/saved_models/experiment{0}".format(str(experiment_id))
        cp_path = cp_dir + "/cp-{epoch:04d}.ckpt"

        if not os.path.isdir(cp_dir):
            os.mkdir(cp_dir)

        if load:
            print("[model.py] loading saved model of experiment", experiment_id)
            self.model = load_model(cp_dir, custom_objects={'knn_loss': self.knn_loss})
        else:
            # init model
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                 save_weights_only=True,
                                                 verbose=10)
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
                

            self.init_model()

            # restore latest checkpoint
            latest_cp = tf.train.latest_checkpoint(cp_dir)
            print("[model.py] latest checkpoint is:", latest_cp)

            if load_cp:
                self.model.load_weights(latest_cp)
            # # print("model summary", self.model.summary())
            # self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=20, epochs=10, callbacks=(epoch_callback()))
            model_history = self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=batch_size, epochs=num_epochs, callbacks=[cp_callback],\
                            validation_data=([val_org_sketches, val_tar_sketches], val_cmb_sketches))
            
            # save the model 
            if save:
                print("[model.py] saving new model of experiment ", experiment_id)
                self.model.save(cp_dir)
            
        # print("resulted loss", self.np_knn_loss(cmb_sketches, params))
        
        # visualize selected pairs
        org_objs = ObjectUtil.accumalitive_stroke3_to_poly(org_sketches)
        tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(tar_sketches)

        val_org_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_org_sketches)
        val_tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_tar_sketches)
        print("first obj len", len(val_org_objs[0]), len(val_org_objs[1]))
        # org_transformed = copy.deepcopy(org_objs)

        # for obj, p in zip(org_objs, params):
        #     obj.transform(RegistrationUtils.obtain_transformation_matrix(p))
        
        # for i in range(len(org_sketches)):
        #     # animation = SketchAnimation([org_objs[i]], [tar_objs[i]])
        #     # print(RegistrationUtils.calc_dissimilarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p), target_dis=False))
        #     # animation.seq_animate_all([params[i]])
        #     org_transformed[i].transform(RegistrationUtils.obtain_transformation_matrix(params[i]))
        print("here")
        # visualize random 20 objects
        if vis_train:
            tag = ""
            vis_dir = "../registrationNN/saved_results/experiment{0}".format(str(experiment_id))
            if not os.path.isdir(vis_dir):
                os.mkdir(vis_dir)
            vis_dir = os.path.join(vis_dir, 'on_training')
            if not os.path.isdir(vis_dir):
                os.mkdir(vis_dir)

            print("[models.py] Saving training visualizations")
            params = self.model.predict((org_sketches, tar_sketches))
            for j in range(10):
                inds = rnd.choices(range(len(org_sketches)), k=5)
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    org_objs[ind].reset()
                    org_objs[ind].visualize(ax=axs[i][0], show=False)
                    tar_objs[ind].visualize(ax=axs[i][1], show=False)
                    tar_objs[ind].visualize(ax=axs[i][2], show=False)
                    org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                    org_objs[ind].visualize(ax=axs[i][2], show=False)
                    org_objs[ind].reset()
                    
                plt.savefig(vis_dir + "/res{0}-{1}.png".format(j, tag))

            train_loss = model_history.history['loss']
            xc = range(num_epochs)
            ax = plt.subplot()
            ax.set_title("Training loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            ax.plot(xc, train_loss)
            plt.savefig(vis_dir + "/epochs-loss.png")

        # visualizing the validation
        if vis_test:
            tag = "without-refinment"
            # predict transformation
            print("predicting")
            params = self.model.predict((val_org_sketches, val_tar_sketches))
            print("resulted average loss without refinment", np.mean(self.np_knn_loss(val_cmb_sketches, params)))
            vis_dir = "../registrationNN/saved_results/experiment{0}/on_testing".format(str(experiment_id))
            if not os.path.isdir(vis_dir):
                os.mkdir(vis_dir)

            print("[models.py] Saving testing visualizations")
            for j in range(10):
                # inds = rnd.choices(range(len(val_org_sketches)), k=5)
                inds = np.concatenate([[0], rnd.choices(range(len(val_org_sketches)), k=4)])
                fig, axs = plt.subplots(len(inds), 3)
                for i, ind in enumerate(inds):
                    val_org_objs[ind].reset()
                    val_org_objs[ind].visualize(ax=axs[i][0], show=False)
                    val_tar_objs[ind].visualize(ax=axs[i][1], show=False)
                    val_tar_objs[ind].visualize(ax=axs[i][2], show=False)
                    val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                    val_org_objs[ind].visualize(ax=axs[i][2], show=False)
                    val_org_objs[ind].reset()

                plt.savefig(vis_dir + "/res{0}-{1}.png".format(j, tag))
            
            if refine_prediction:
                tag = "with-refinement"
                refine_epochs = 10
                refine_history = self.model.fit(x=[val_org_sketches, val_tar_sketches], y=val_cmb_sketches, batch_size=batch_size, epochs=refine_epochs)
                # visualize loss
                ax = plt.subplot()
                ax.set_title("Testing loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("loss")
                ax.plot(range(refine_epochs), refine_history.history['loss'])
                plt.savefig(vis_dir + "/refine-epochs-loss{0}.png".format(tag))
                params = self.model.predict((val_org_sketches, val_tar_sketches))
                print("resulted average loss after refinment", np.mean(self.np_knn_loss(val_cmb_sketches, params)))

                # TEST: uncomment
                print("[models.py] Saving testing visualizations")
                for j in range(10):
                    # inds = rnd.choices(range(len(val_org_sketches)), k=5)
                    inds = np.concatenate([[0], rnd.choices(range(len(val_org_sketches)), k=4)])
                    fig, axs = plt.subplots(len(inds), 3)
                    for i, ind in enumerate(inds):
                        val_org_objs[ind].visualize(ax=axs[i][0], show=False)
                        val_tar_objs[ind].visualize(ax=axs[i][1], show=False)
                        val_tar_objs[ind].visualize(ax=axs[i][2], show=False)
                        val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                        val_org_objs[ind].visualize(ax=axs[i][2], show=False)
                        val_org_objs[ind].reset()

                    plt.savefig(vis_dir + "/res{0}-{1}.png".format(j, tag))
            
            if iter_refine_prediction:
                print("[registration_model] predicting with iterative refinement")
                tag = "with-iter-refinement"
                refine_iter = 10
                N = 7
                # choose random k indices
                # TEST: for now always including an identical object in the visualization
                inds = np.concatenate([[0], rnd.choices(range(len(val_org_sketches)), k=N-1)])
                for i in range(refine_iter):
                    params = self.model.predict((val_org_sketches, val_tar_sketches))
                    print("resulted average loss with iterative refinment at iter {0}: {1}".format(i, np.mean(self.np_knn_loss(val_cmb_sketches, params))))
                    # visualize first 5 sketches
                    fig, axs = plt.subplots(N, 3)
                    for j, ind in enumerate(inds):
                        print(params[ind])
                        val_org_objs[ind].visualize(ax=axs[j][0], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][1], show=False)
                        val_tar_objs[ind].visualize(ax=axs[j][2], show=False)
                        val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                        val_org_objs[ind].visualize(ax=axs[j][2], show=False)
                    
                    plt.savefig(vis_dir + "/iter{0}-{1}.png".format(i, tag))

                    # transform all validation sketches
                    for j in range(len(val_org_objs)):
                        if j in inds:
                            continue
                        val_org_objs[j].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))


                    # obtain stroke 3 representation
                    val_org_sketches = ObjectUtil.poly_to_accumulative_stroke3(val_org_objs, red_rdp=False, normalize=False)
                    val_org_sketches = self._pad_sketches(val_org_sketches, maxlen=128)
                    val_org_sketches = np.expand_dims(val_org_sketches, axis=-1)
                    val_cmb_sketches = np.stack((val_org_sketches, val_tar_sketches), axis=1)
                    # obtain objs from stroke 3 represtentation
                    val_org_objs = ObjectUtil.accumalitive_stroke3_to_poly(val_org_sketches)



                # visualize loss
                ax = plt.subplot()
                ax.set_title("Testing loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("loss")
                ax.plot(range(refine_epochs), refine_history.history['loss'])
                plt.savefig(vis_dir + "/refine-epochs-loss{0}.png".format(tag))
                params = self.model.predict((val_org_sketches, val_tar_sketches))
                print("resulted average loss after refinment", np.mean(self.np_knn_loss(val_cmb_sketches, params)))

                print("[registration_model] Saving testing visualizations")
                for j in range(10):
                    inds = rnd.choices(range(len(val_org_sketches)), k=5)
                    fig, axs = plt.subplots(len(inds), 3)
                    for i, ind in enumerate(inds):
                        val_org_objs[ind].reset()
                        val_org_objs[ind].visualize(ax=axs[i][0], show=False)
                        val_tar_objs[ind].visualize(ax=axs[i][1], show=False)
                        val_tar_objs[ind].visualize(ax=axs[i][2], show=False)
                        val_org_objs[ind].transform(RegistrationUtils.obtain_transformation_matrix(params[ind]))
                        val_org_objs[ind].visualize(ax=axs[i][2], show=False)

                    plt.savefig(vis_dir + "/res{0}-{1}.png".format(j, tag))
            # visualize loss
            val_loss  = model_history.history['val_loss']
            xc = range(num_epochs)
            ax = plt.subplot()
            ax.set_title("Testing loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("loss")
            ax.plot(xc, val_loss)
            plt.savefig(vis_dir + "/epochs-loss{0}.png".format(str(datetime.now())))
        # tf.keras.utils.plot_model(self.model, to_file='model.png', show_layer_names=True, rankdir='TB', show_shapes=True)
        
