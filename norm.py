#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 20:26:53 2018

@author: liuhuihui
"""

import numpy as np


tagFeature=np.load('/home/liuhuihui/NEWwork/data/features/tag/tagFeature.npy')
tagFeature=tagFeature.astype(np.float64)
sum_tag=np.sum(tagFeature,axis=1)
higt,width=tagFeature.shape
for i in range(higt):
    print i
    if sum_tag[i]!=0:
        tagFeature[i,:]=tagFeature[i,:]/sum_tag[i]
np.save('/home/liuhuihui/NEWwork/data/features/tag/tagFeature_norm.npy',tagFeature)

userFeature=np.load('/home/liuhuihui/NEWwork/data/features/user/userFeature.npy')
userFeature=userFeature.astype(np.float64)
sum_user=np.sum(userFeature,axis=1)
higt,width=userFeature.shape
for i in range(higt):
    print i
    if sum_user[i]!=0:
        userFeature[i,:]=userFeature[i,:]/sum_user[i]
np.save('/home/liuhuihui/NEWwork/data/features/user/userFeature_norm.npy',userFeature)

groupFeature=np.load('/home/liuhuihui/NEWwork/data/features/group/groupFeature.npy')
groupFeature=groupFeature.astype(np.float64)
sum_group=np.sum(groupFeature,axis=1)
higt,width=groupFeature.shape
for i in range(higt):
    print i
    if sum_group[i]!=0:
        groupFeature[i,:]=groupFeature[i,:]/sum_group[i]
np.save('/home/liuhuihui/NEWwork/data/features/group/groupFeature_norm.npy',groupFeature)

sum_group=np.sum(groupFeature,axis=1)
