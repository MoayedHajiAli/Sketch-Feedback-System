import keras.backend as K
from keras.layers import Input, Dense, concatenate, Conv2D, Reshape, Flatten
from keras.losses import mse
from keras import Model
from utils.ObjectUtil import ObjectUtil
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf

class registration_model:
  
    def knn_loss(self, org0, tar0):
        org0 = org0[:, :, :2, 0]
        tar0 = tar0[:, :, :2, 0]
        def loss(sketches, p):
            org = sketches[:, 0, :, :2, 0]
            tar = sketches[:, 1, :, :2, 0]
            # # obtain transformation matrix parameters
            t = []
            t.append(p[:, 0] * (K.cos(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) - p[:, 4] * K.sin(p[:, 2])))
            t.append(p[:, 0] * (p[:, 3] * K.cos(p[:, 2]) - K.sin(p[:, 2])))
            t.append(p[:, 5])
            t.append(p[:, 1] * (K.sin(p[:, 2]) * (1 + p[:, 3] * p[:, 4]) + p[:, 4] * K.cos(p[:, 2])))
            t.append(p[:, 1] * (p[:, 3] * K.sin(p[:, 2]) + K.cos(p[:, 2])))
            t.append(p[:, 6])
            t = K.expand_dims(t, -1)
            # apply transformation on all points in original 
            # org (None, 128, 3) , p(None,)
            org1 = org[:, :, 0] * t[0]
            org1 = org[:, :, 0] * t[0] + org[:, :, 1] * t[1] + t[2]
            org2 = org[:, :, 0] * t[3] + org[:, :, 1] * t[4] + t[5]
            # org1, org2 = org[:, :, 0] * p[:,0] + org[:, :, 1] * p[:,1] + p[:,2],  org[:, :, 0] * p[:,3] + org[:, :, 1] * p[:,4] + p[:,5]
            org1 = K.expand_dims(org1, 2)
            org2 = K.expand_dims(org2, 2)
            org_cmb = tf.concat([org1, org2], axis=-1)
            org_cmb = K.expand_dims(org_cmb, 1)
            tar_cmb = K.expand_dims(tar, 2)
            dif = org_cmb - tar_cmb
            sm = K.sum(dif ** 2, axis=-1)
            mn = K.min(sm, axis=-1)
            return K.sum(mn, axis=1)
            
        return loss            

    def init_model(self):
        # build the model with stroke-3 format
        org_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)
        tar_inputs = Input(shape=(128, 3, 1), dtype=tf.float32)

        # feature extraction layers
        org_fe_layer0 = Conv2D(1, (3, 3), 1)(org_inputs)
        org_fe_layer0 = Reshape((126,)) (org_fe_layer0)
        org_fe_layer1= Dense(126, activation='linear')(org_fe_layer0)
        org_fe_layer2= Dense(64, activation='relu')(org_fe_layer1)
        org_fe_layer3= Dense(32, activation='linear')(org_fe_layer2)
        org_fe = Model(org_inputs, org_fe_layer3)

        tar_fe_layer0 = Conv2D(1, (3, 3), 1)(tar_inputs)
        tar_fe_layer0 = Reshape((126, )) (tar_fe_layer0)
        tar_fe_layer1= Dense(126, activation='linear')(tar_fe_layer0)
        tar_fe_layer2= Dense(64, activation='relu')(tar_fe_layer1)
        tar_fe_layer3= Dense(32, activation='linear')(tar_fe_layer2)
        tar_fe = Model(tar_inputs, tar_fe_layer3)

        merged_fe = concatenate([org_fe.output, tar_fe.output])
        # original and target extracted features
        layer1 = Dense(32, activation='relu')(merged_fe)
        params = Dense(7, activation="linear")(layer1)

        self.model = Model(inputs=[org_fe.input, tar_fe.input], outputs=params)
        self.model.compile(loss=self.knn_loss(org_inputs, tar_inputs), optimizer="adam", metrics=['accuracy'])

    def __init__(self, train_sketches):
        tf.config.RUN_FUNCTIONS_EAGERLY = False
        # init model
        self.init_model()

        # convert train sketches into stroke-3 format
        train_sketches = ObjectUtil.poly_to_stroke3(train_sketches)

        # add padding
        train_sketches = pad_sequences(train_sketches, maxlen=128)

        # add every pair of objects to the training set
        org_sketches, tar_sketches = [], []
        for i in range(len(train_sketches)):
            for j in range(len(train_sketches)):
                org_sketches.append(train_sketches[i])
                tar_sketches.append(train_sketches[j])
        
        org_sketches = np.float32(np.expand_dims(org_sketches, -1))
        tar_sketches = np.float32(np.expand_dims(tar_sketches, -1))
        cmb_sketches = np.stack((org_sketches, tar_sketches), axis=1)
        self.model.fit(x=[org_sketches, tar_sketches], y=cmb_sketches, batch_size=20, epochs=1000)
        tf.keras.utils.plot_model(self.model, to_file='model.png', show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)