import keras.backend as K
from keras.layers import Input, Dense, Concatenate
from keras import Model
from utils.ObjectUtil import ObjectUtil
from keras_preprocessing.sequence import pad_sequences
import numpy as np

class registration_model:

    def knn_loss(self, org, tar):

        def loss(p, _):
            # obtain transformation matrix parameters
            print(p[0])
            t = np.zeros(6)
            t[0] = p[0] * (K.cos(p[2]) * (1 + p[3] * p[4]) - p[4] * K.sin(p[2]))
            t[1] = p[0] * (p[3] * K.cos(p[2]) - K.sin(p[2])) 
            t[2] = p[5]
            t[3] = p[1] * (K.sin(p[2]) * (1 + p[3] * p[4]) + p[4] * K.cos(p[2]))
            t[4] = p[1] * (p[3] * K.sin(p[2]) + K.cos(p[2]))
            t[5] = p[6]
            # apply transformation on all points in original
            org[:, 0], org[:, 1] = org[:, 0] * t[0] + org[:, 1] * t[1] + t[2],  org[:, 0] * t[3] + org[:, 1] * t[4] + t[5]

            sum = 0
            for p in org:
                mn = 1000000
                for q in tar:
                    mn = K.min([mn, (p[0]-q[0]) ** 2 + (p[1]-q[1]) ** 2])
                sum += mn
            
            return sum     
        return loss            


    def init_model(self):
        # build the model with stroke-3 format
        org_inputs = Input(shape=(128,3))
        tar_inputs = Input(shape=(128,3))

        # feature extraction layers
        org_fe_layer1= Dense(128, activation='linear')(org_inputs)
        org_fe_layer2= Dense(64, activation='relu')(org_fe_layer1)
        org_fe_layer3= Dense(32, activation='linear')(org_fe_layer2)
        org_fe = Model(org_inputs, org_fe_layer3)

        tar_fe_layer1= Dense(128, activation='linear')(tar_inputs)
        tar_fe_layer2= Dense(64, activation='relu')(tar_fe_layer1)
        tar_fe_layer3= Dense(32, activation='linear')(tar_fe_layer2)
        tar_fe = Model(tar_inputs, tar_fe_layer3)

        merged_fe = Concatenate()(([org_fe.output, tar_fe.output]))
        # original and target extracted features
        layer1 = Dense(32, activation='relu')(merged_fe)
        params = Dense(7, activation="linear")(layer1)

        self.model = Model([org_fe.input, tar_fe.input], params)
        self.model.compile(loss=self.knn_loss(org_inputs, tar_inputs), optimizer="adam", metrics=['accuracy'])

    def __init__(self, train_sketches):
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


        self.model.fit([np.asarray(org_sketches), np.asarray(tar_sketches)], np.zeros(len(org_sketches)), batch_size=20, epochs=20)