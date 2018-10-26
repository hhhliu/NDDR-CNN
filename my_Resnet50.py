#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:50:13 2018

@author: liuhuihui
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.utils import plot_model
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def identity_block(input_tensor, kernel_size, filters, stage, block, social):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch' + social

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), social=None):
   
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch' + social
    bn_name_base = 'bn' + str(stage) + block + '_branch' + social

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def merge(tag, user, group,n, filters):
    name_tag = 'NDDR'+n+'_tag'
    name_user = 'NDDR'+n+'_user'
    name_group = 'NDDR'+n+'_group'
    pool_merge=concatenate([tag, user, group])   
    pool_bn = BatchNormalization()(pool_merge)  
    tag = Conv2D(filters, (1, 1), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01), name=name_tag)(pool_bn)  
    user = Conv2D(filters, (1, 1), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01), name=name_user)(pool_bn)
    group = Conv2D(filters, (1, 1), strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01), name=name_group)(pool_bn)
    
    return tag, user, group
    
def mysoftmax(x, axis=-1):
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return e
                    
    
def myResnet(input_tensor):
    
    bn_axis = 3
    
    tag = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1_tag')(input_tensor)
    tag = BatchNormalization(axis=bn_axis, name='bn_conv1_tag')(tag)
    tag = Activation('relu')(tag)
    tag = MaxPooling2D((3, 3), strides=(2, 2))(tag)
    
    user = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1_user')(input_tensor)
    user = BatchNormalization(axis=bn_axis, name='bn_conv1_user')(user)
    user = Activation('relu')(user)
    user = MaxPooling2D((3, 3), strides=(2, 2))(user)
    
    group = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1_group')(input_tensor)
    group = BatchNormalization(axis=bn_axis, name='bn_conv1_group')(group)
    group = Activation('relu')(group)
    group = MaxPooling2D((3, 3), strides=(2, 2))(group)
    
    tag, user, group = merge(tag, user, group,'1', 64)
    
    tag = conv_block(tag, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social= 'tag')
    tag = identity_block(tag, 3, [64, 64, 256], stage=2, block='b', social= 'tag')
    tag = identity_block(tag, 3, [64, 64, 256], stage=2, block='c', social= 'tag')
    
    user = conv_block(user, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social= 'user')
    user = identity_block(user, 3, [64, 64, 256], stage=2, block='b', social= 'user')
    user = identity_block(user, 3, [64, 64, 256], stage=2, block='c', social= 'user')
    
    group = conv_block(group, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), social= 'group')
    group = identity_block(group, 3, [64, 64, 256], stage=2, block='b', social= 'group')
    group = identity_block(group, 3, [64, 64, 256], stage=2, block='c', social= 'group')
    
    tag, user, group = merge(tag, user, group,'2', 256)
    
    tag = conv_block(tag, 3, [128, 128, 512], stage=3, block='a', social= 'tag')
    tag = identity_block(tag, 3, [128, 128, 512], stage=3, block='b', social= 'tag')
    tag = identity_block(tag, 3, [128, 128, 512], stage=3, block='c', social= 'tag')
    tag = identity_block(tag, 3, [128, 128, 512], stage=3, block='d', social= 'tag')
    
    user = conv_block(user, 3, [128, 128, 512], stage=3, block='a', social= 'user')
    user = identity_block(user, 3, [128, 128, 512], stage=3, block='b', social= 'user')
    user = identity_block(user, 3, [128, 128, 512], stage=3, block='c', social= 'user')
    user = identity_block(user, 3, [128, 128, 512], stage=3, block='d', social= 'user')
    
    group = conv_block(group, 3, [128, 128, 512], stage=3, block='a', social= 'group')
    group = identity_block(group, 3, [128, 128, 512], stage=3, block='b', social= 'group')
    group = identity_block(group, 3, [128, 128, 512], stage=3, block='c', social= 'group')
    group = identity_block(group, 3, [128, 128, 512], stage=3, block='d', social= 'group')
    
    tag, user, group = merge(tag, user, group,'3', 512)
    
    tag = conv_block(tag, 3, [256, 256, 1024], stage=4, block='a', social= 'tag')
    tag = identity_block(tag, 3, [256, 256, 1024], stage=4, block='b', social= 'tag')
    tag = identity_block(tag, 3, [256, 256, 1024], stage=4, block='c', social= 'tag')
    tag = identity_block(tag, 3, [256, 256, 1024], stage=4, block='d', social= 'tag')
    tag = identity_block(tag, 3, [256, 256, 1024], stage=4, block='e', social= 'tag')
    tag = identity_block(tag, 3, [256, 256, 1024], stage=4, block='f', social= 'tag')
    
    user = conv_block(user, 3, [256, 256, 1024], stage=4, block='a', social= 'user')
    user = identity_block(user, 3, [256, 256, 1024], stage=4, block='b', social= 'user')
    user = identity_block(user, 3, [256, 256, 1024], stage=4, block='c', social= 'user')
    user = identity_block(user, 3, [256, 256, 1024], stage=4, block='d', social= 'user')
    user = identity_block(user, 3, [256, 256, 1024], stage=4, block='e', social= 'user')
    user = identity_block(user, 3, [256, 256, 1024], stage=4, block='f', social= 'user')
    
    group = conv_block(group, 3, [256, 256, 1024], stage=4, block='a', social= 'group')
    group = identity_block(group, 3, [256, 256, 1024], stage=4, block='b', social= 'group')
    group = identity_block(group, 3, [256, 256, 1024], stage=4, block='c', social= 'group')
    group = identity_block(group, 3, [256, 256, 1024], stage=4, block='d', social= 'group')
    group = identity_block(group, 3, [256, 256, 1024], stage=4, block='e', social= 'group')
    group = identity_block(group, 3, [256, 256, 1024], stage=4, block='f', social= 'group')
    
    tag, user, group = merge(tag, user, group,'4', 1024)
    
    tag = conv_block(tag, 3, [512, 512, 2048], stage=5, block='a', social= 'tag')
    tag = identity_block(tag, 3, [512, 512, 2048], stage=5, block='b', social= 'tag')
    tag = identity_block(tag, 3, [512, 512, 2048], stage=5, block='c', social= 'tag')
    
    user = conv_block(user, 3, [512, 512, 2048], stage=5, block='a', social= 'user')
    user = identity_block(user, 3, [512, 512, 2048], stage=5, block='b', social= 'user')
    user = identity_block(user, 3, [512, 512, 2048], stage=5, block='c', social= 'user')
    
    group = conv_block(group, 3, [512, 512, 2048], stage=5, block='a', social= 'group')
    group = identity_block(group, 3, [512, 512, 2048], stage=5, block='b', social= 'group')
    group = identity_block(group, 3, [512, 512, 2048], stage=5, block='c', social= 'group')
    
    tag, user, group = merge(tag, user, group,'5', 2048)
    
    tag = AveragePooling2D((7, 7), name='avg_pool_tag')(tag)
    tag = Flatten(name='flatten_tag')(tag)
    tag = Dense(474, activation=mysoftmax, name='predictions_tag')(tag)
    
    user = AveragePooling2D((7, 7), name='avg_pool_user')(user)
    user = Flatten(name='flatten_user')(user)
    user = Dense(336, activation=mysoftmax, name='predictions_user')(user)
    
    group = AveragePooling2D((7, 7), name='avg_pool_group')(group)
    group = Flatten(name='flatten_group')(group)
    group = Dense(335, activation=mysoftmax, name='predictions_group')(group) 
    
    model=Model(outputs=(tag,user,group),inputs=input_tensor)
    
    return model
    

if __name__ == '__main__':

    input_shape = (224, 224, 3)
    inputTensor = Input(shape=input_shape)
   
    model= myResnet(inputTensor)
    
    model.summary()
    plot_model(model,to_file='/home/liuhuihui/NEWwork/resnet_model.png')