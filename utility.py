# -*- coding: utf-8 -*-

import numpy as np

def softmax_sample(prob):
    s = np.zeros(prob.shape) 
    for n in range(s.shape[0]):
        i = np.random.choice(s.shape[1],1,p=prob[n,:])
        s[n,i] = 1
    return s