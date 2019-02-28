# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:37:45 2019

@author: Administrator
"""

import theano
import theano.tensor as T
import numpy as np

def sample_with_softmax(logits, size):
# logits为输入数据
# size为采样数
    pro = T.nnet.softmax(logits)
    return np.random.choice(len(logits), size, p=pro)



z = sample_with_softmax(T.fvector([[1,2,3,4,5]]),3)