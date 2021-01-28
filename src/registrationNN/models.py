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

    def __init__(self, train_sketches):

        # convert train sketches into stroke-3 format
        train_sketches = ObjectUtil.poly_to_accumulative_stroke3(train_sketches)

        # add padding
        train_sketches = self._pad_sketches(train_sketches, maxlen=128)
        # train_sketches = pad_sequences(train_sketches, maxlen=128)

        # add every pair of objects to the training set
        org_sketches, tar_sketches = [], []
        for i in range(len(train_sketches)):
            for j in range(len(train_sketches)):
                org_sketches.append(np.array(train_sketches[i]))
                tar_sketches.append(np.array(train_sketches[j]))
        
        # tar_sketches = np.array(train_sketches[0:10])
        # org_sketches = np.array(train_sketches[0:10])
        # org_sketches = org_sketches[:20]
        # tar_sketches = tar_sketches[:20]
        # test tranlation
        # org_sketches = train_sketches[0:5]
        # tar_sketches = train_sketches[0:5]
        org_sketches = np.expand_dims(org_sketches, axis=-1)
        tar_sketches = np.expand_dims(tar_sketches, axis=-1)
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)

        experiment_id = 1
        batch_size = 20
        load = False
        save = True
        cp_dir = "../registrationNN/saved_models/experiment{0}".format(str(experiment_id))
        cp_path = cp_dir + "/cp-{epoch:04d}.ckpt"

        if not os.path.isdir(cp_dir):
            os.mkdir(cp_dir)

        if load:
            self.model = load_model(cp_dir, custom_objects={'knn_loss': self.knn_loss})
        else:
            # init model
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                 save_weights_only=True,
                                                 verbose=10,
                                                 save_freq=10*batch_size)
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
            print(latest_cp)
            self.model.load_weights(latest_cp)
            # print("model summary", self.model.summary())
            # self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=20, epochs=10, callbacks=(epoch_callback()))
            self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=batch_size, epochs=100, callbacks=[cp_callback])
            # save the model 
            if save:
                self.model.save("saved_models/experiment" + str(experiment_id))
            
        # predict transformation
        params = self.model.predict((org_sketches, tar_sketches))
        print("params", params)


        print("resulted loss", self.np_knn_loss(cmb_sketches, params))
        # visualize selected pairs
        org_objs = ObjectUtil.accumalitive_stroke3_to_poly(org_sketches)
        tar_objs = ObjectUtil.accumalitive_stroke3_to_poly(tar_sketches)

        # for obj, p in zip(org_objs, params):
        #     obj.transform(RegistrationUtils.obtain_transformation_matrix(p))
        
        for i in range(len(org_sketches)):
            # animation = SketchAnimation([org_objs[i]], [tar_objs[i]])
            # print(RegistrationUtils.calc_dissimilarity(obj1, obj2, RegistrationUtils.obtain_transformation_matrix(p), target_dis=False))
            # animation.seq_animate_all([params[i]])
            org_objs[i].transform(RegistrationUtils.obtain_transformation_matrix(params[i]))
        
        # visualize random 20 objects
        for j in range(5):
            inds = rnd.choices(range(len(org_sketches)), k=16)
            fig, axs = plt.subplots(4, 4)
            for i, ind in enumerate(inds):
                org_objs[ind].visualize(ax=axs[int(i/4)][int(i%4)], show=False)
                tar_objs[ind].visualize(ax=axs[int(i/4)][int(i%4)], show=False)
                axs[int(i/4)][int(i%4)].set_axis_off()
            
            plt.savefig(cp_dir + "_res{0}.png".format(j))


        tf.keras.utils.plot_model(self.model, to_file='model.png', show_layer_names=True, rankdir='TB', show_shapes=True)
        