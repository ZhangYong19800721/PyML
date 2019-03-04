# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:37:45 2019

@author: Administrator
"""

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams()

A = np.choose([[0],[1],[0]],np.eye(10))
