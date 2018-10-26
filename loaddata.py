#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 09:45:04 2018

@author: liuhuihui
"""

import numpy as np
import h5py
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.applications.resnet50 import preprocess_input

outsize=256
n_train=137959.0
train_shape=(137959,256,256,3)
mean=np.zeros((256,256,3),np.float32)

image_dir = '/home/liuhuihui/Social_NUSWIDE/validated_photos/'
train_file = h5py.File('/home/liuhuihui/NEWwork/data/train_Social2.hdf5', 'w')
dset = train_file.create_dataset("images", train_shape, np.uint8)   
f=open('/home/liuhuihui/NEWwork/data/preprocessed/group/iamgeID_groupfilter.txt','r')
lines = f.readlines()
i=0
for line in lines:
    if (i + 1) % 1000 == 0:
        print 'Train data: {}/{}'.format(i + 1, 137959)
    line=line.strip('\n') 
    print line
    
    
'''    
    addr = image_dir+line+'.jpg'
    img = image.load_img(addr,target_size=(256,256))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    dset[i, ...] = x
    mean += x/n_train
    i+=1

train_file.create_dataset("mean", data = mean)
train_file.close()
'''

