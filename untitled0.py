# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:37:45 2019

@author: Administrator
"""

import theano
import theano.tensor as T
import numpy as np

A = T.fmatrix('A')
D = theano.tensor.basic.choose(1, A)
f = theano.function([A],[D])

x = np.array([[0.1,0.1,0.8],[0.2,0.4,0.4],[0.7,0.2,0.1]],dtype='float32')
d = f(x)

print(d)