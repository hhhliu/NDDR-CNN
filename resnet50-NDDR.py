# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:26:37 2018

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
from keras.layers.merge import concatenate
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras.backend as K
from keras.utils import plot_model
from keras.layers import Merge
from my_Resnet50 import myResnet
from LR_SGD import LR_SGD
from sklearn.metrics import roc_curve,roc_auc_score
from keras.utils import multi_gpu_model
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

def generate_sequences(n_batches, images, labels_tag,labels_user,labels_group, mean, idxs):
    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                                  
            batch_length=len(batch_idxs)
            X = np.zeros((batch_length,224,224,3),np.float32)            
                     
            Y_tag = labels_tag[batch_idxs]
            Y_user = labels_user[batch_idxs]
            Y_group = labels_group[batch_idxs]
          
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
            
            yield X,[Y_tag,Y_user,Y_group]
            
def myloss(y_true,y_pred):
        y_pred=K.clip(y_pred,K.epsilon(),1 - K.epsilon())
        return K.mean(K.sum(- y_true * K.log(y_pred), axis=1))
        
if __name__ == '__main__':
    
    input_shape = (224, 224, 3)
    inputTensor = Input(shape=input_shape,name = 'image_input')
    model = myResnet(inputTensor)
    
    model.load_weights('/home/liuhuihui/NEWwork/initialize/weights/resnet50-tag-mysoftmax.h5',by_name=True)
    model.load_weights('/home/liuhuihui/NEWwork/initialize/weights/resnet50-user-mysoftmax.h5',by_name=True)
    model.load_weights('/home/liuhuihui/NEWwork/initialize/weights/resnet50-group-mysoftmax.h5',by_name=True)
    
    model.summary()
  
    LR_mult_dict = {}
    LR_mult_dict['NDDR1_tag']=10
    LR_mult_dict['NDDR1_user']=10
    LR_mult_dict['NDDR1_group']=10
    
    LR_mult_dict['NDDR2_tag']=10
    LR_mult_dict['NDDR2_user']=10
    LR_mult_dict['NDDR2_group']=10
    
    LR_mult_dict['NDDR3_tag']=10
    LR_mult_dict['NDDR3_user']=10
    LR_mult_dict['NDDR3_group']=10
    
    LR_mult_dict['NDDR4_tag']=10
    LR_mult_dict['NDDR4_user']=10
    LR_mult_dict['NDDR4_group']=10
    
    LR_mult_dict['NDDR5_tag']=10
    LR_mult_dict['NDDR5_user']=10
    LR_mult_dict['NDDR5_group']=10

    base_lr = 0.0001
    momentum = 0.9
    optimizer = LR_SGD(lr=base_lr, momentum=momentum, decay=1e-6, nesterov=True,multipliers = LR_mult_dict)
    rss_model=multi_gpu_model(model,gpus=3)
    rss_model.compile(optimizer = optimizer,loss = myloss)
    
    batch_size = 32
    nb_epoch = 200
    validation_ratio = 0.15

    labels_tag = np.load('/home/liuhuihui/NEWwork/data/features/tag/tagFeature_norm.npy')
    labels_user = np.load('/home/liuhuihui/NEWwork/data/features/user/userFeature_norm.npy')
    labels_group = np.load('/home/liuhuihui/NEWwork/data/features/group/groupFeature_norm.npy')

    path_train = '/home/liuhuihui/NEWwork/data/train_Social.hdf5'
    with h5py.File(path_train, 'r') as train_file:
        images = train_file['images']
        mean = train_file['mean'][...]
    
        idxs = range(len(images))
        train_idxs = idxs[: int(len(images) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images) * (1 - validation_ratio)) :]

        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches, images, labels_tag,labels_user,labels_group, mean, train_idxs)

        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_validation_batches, images, labels_tag,labels_user,labels_group, mean,validation_idxs)
        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/NEWwork/weights/resnet50-nddr-mysoftmax.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
        
        history = rss_model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])  
  