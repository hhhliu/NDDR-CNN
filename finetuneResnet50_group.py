# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:40:07 2018

@author: liuhuihui
"""

from keras import regularizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,LearningRateScheduler
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense
from keras.applications.resnet50 import ResNet50
#from resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
import h5py
import random
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras.backend as K
from keras.layers import Merge
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def generate_sequences(n_batches, images, labels_group, mean, idxs):
    # generate batches of samples
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)                              
            Y_group = labels_group[batch_idxs,:]
            
            count=0
            # for every image of a batch
            for i in batch_idxs:
                xx=images[i,...].astype(np.float32)
                xx-=mean
                offset_x=random.randint(0,31)
                offset_y=random.randint(0,31)
                xx=xx[offset_x:offset_x+224,offset_y:offset_y+224,:]
                
                X[count,...]=xx
                count+=1    
            
            yield X,Y_group


if __name__ == '__main__':
    
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    resnet50 = ResNet50(include_top=False, weights=None,input_tensor=input_tensor)
    resnet50.load_weights('/home/liuhuihui/ME/data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    output_resnet_conv = resnet50(input_tensor)
    x = Flatten(name='flatten')(output_resnet_conv)
    model_group=Dense(335, activation='softmax',name='group')(x)
  
    model=Model(outputs=model_group,inputs=input_tensor)   
    model.summary()
    
    def myloss(y_true,y_pred):
        #Avoid divide by 0
        #y_pred=K.clip(y_pred,K.epsilon(),1 - K.epsilon())
        #multi-task loss
        return -K.mean(K.sum(y_true * K.log(y_pred) ,axis=1))
        
    sgd = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
    
    #model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy')
    model.compile(optimizer=sgd,loss='categorical_crossentropy')

    batch_size = 32
    nb_epoch = 500
    validation_ratio = 0.2

    path_train = '/home/liuhuihui/NEWwork/data/train_Social.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']

        labels_group = np.load('/home/liuhuihui/NEWwork/data/features/group/groupFeature.npy')
        mean = train_file['mean'][...]
    
        idxs = range(len(images))
        train_idxs = idxs[: int(len(images) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images) * (1 - validation_ratio)) :]

        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches, images, labels_group, mean, train_idxs)

        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_validation_batches, images, labels_group, mean,validation_idxs)
        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/NEWwork/initialize/weights/group.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])     
    