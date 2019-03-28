# -*- coding: utf-8 -*-
"""
    长短期记忆网络（LSTM，Long Short Term Memory）
    基于theano实现，使用GPU训练
"""

import theano
import theano.tensor as T
import numpy as np
import collections
from optimizer import minimize
from dataset import mnist
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams()

class LSTM(object):
    # 构造函数
    # 输入：
    #      visual_dim (int) 可见神经元的个数
    #      hidden_dim (int) 隐藏神经元的个数
    def __init__(self, visual_dim, hidden_dim):
        pass
        self.parameters = collections.OrderedDict()  # 模型的参数集合
