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
    #      state_dim (int)  状态神经元的个数
    def __init__(self, visual_dim, hidden_dim):
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.parameters = collections.OrderedDict()  # 模型的参数集合

    # 初始化模型参数
    # 输入：
    #     Wf（numpy ndarray，None）遗忘门的权值矩阵，默认为None
    #     Wi（numpy ndarray，None）选择门的权值矩阵，默认为None
    #     Wc（numpy ndarray，None）更新门的权值矩阵，默认为None
    #     Wo（numpy ndarray，None）输出门的权值矩阵，默认为None
    #     Bf（numpy ndarray，None）遗忘门的偏置，默认为None
    #     Bi（numpy ndarray，None）选择门的偏置，默认为None
    #     Bc（numpy ndarray，None）更新门的偏置，默认为None
    #     Bo（numpy ndarray，None）输出门的偏置，默认为None
    def initialize_parameters(self, Wf=None, Wi=None, Wc=None, Wo=None, Bf=None, Bi=None, Bc=None, Bo=None):
        if Wf is not None:  # 遗忘门的权值矩阵
            assert Wf.shape[0] == self.visual_dim + self.hidden_dim
            assert Wf.shape[1] == self.hidden_dim
        else:
            Wf = np.asarray(0.1 * np.random.randn(self.visual_dim + self.hidden_dim, self.hidden_dim),
                            dtype=theano.config.floatX)
        self.parameters['Wf'] = theano.shared(Wf, name='Wf')

        if Wi is not None:  # 选择门的权值矩阵
            assert Wi.shape[0] == self.visual_dim + self.hidden_dim
            assert Wi.shape[1] == self.hidden_dim
        else:
            Wi = np.asarray(0.1 * np.random.randn(self.visual_dim + self.hidden_dim, self.hidden_dim),
                            dtype=theano.config.floatX)
        self.parameters['Wi'] = theano.shared(Wi, name='Wi')

        if Wc is not None:  # 更新门的权值矩阵
            assert Wc.shape[0] == self.visual_dim + self.hidden_dim
            assert Wc.shape[1] == self.hidden_dim
        else:
            Wc = np.asarray(0.1 * np.random.randn(self.visual_dim + self.hidden_dim, self.hidden_dim),
                            dtype=theano.config.floatX)
        self.parameters['Wc'] = theano.shared(Wc, name='Wc')

        if Wo is not None:  # 输出门的权值矩阵
            assert Wo.shape[0] == self.visual_dim + self.hidden_dim
            assert Wo.shape[1] == self.hidden_dim
        else:
            Wo = np.asarray(0.1 * np.random.randn(self.visual_dim + self.hidden_dim, self.hidden_dim),
                            dtype=theano.config.floatX)
        self.parameters['Wo'] = theano.shared(Wo, name='Wo')

        if Bf is not None:  # 遗忘门的偏置
            assert Bf.shape[0] == self.hidden_dim
        else:
            Bf = np.asarray(np.zeros(self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Bf'] = theano.shared(Bf, name='Bf')

        if Bi is not None:  # 选择门的偏置
            assert Bi.shape[0] == self.hidden_dim
        else:
            Bi = np.asarray(np.zeros(self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Bi'] = theano.shared(Bi, name='Bi')

        if Bc is not None:  # 更新门的偏置
            assert Bc.shape[0] == self.hidden_dim
        else:
            Bc = np.asarray(np.zeros(self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Bc'] = theano.shared(Bc, name='Bc')

        if Bo is not None:  # 输出门的偏置
            assert Bo.shape[0] == self.hidden_dim
        else:
            Bo = np.asarray(np.zeros(self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Bo'] = theano.shared(Bo, name='Bo')

    # 初始化模型
    def initialize_model(self):
        # step函数
        # 输入：
        #  x 表示当前时刻X状态，即x(t)
        # _h 表示上一时刻H状态，即h(t-1)
        # _c 表示上一时刻C状态，即C(t-1)
        def step(x, _h, _c):
            P = self.parameters
            f = T.nnet.sigmoid(T.concatenate([_h, x], axis=1).dot(P['Wf']) + P['Bf'])
            i = T.nnet.sigmoid(T.concatenate([_h, x], axis=1).dot(P['Wi']) + P['Bi'])
            c = T.tanh(T.concatenate([_h, x], axis=1).dot(P['Wc']) + P['Bc'])
            c = f * _c + i * c
            o = T.nnet.sigmoid(T.concatenate([_h, x], axis=1).dot(P['Wo']) + P['Bo'])
            h = o * T.tanh(c)
            return h, c

        X = theano.fmatrix('X')  # 行代表time_step，列代表数据样本
        step_num = X.shape[0]
        sample_num = X.shape[1]
        output, updates = theano.scan(step,
                                      sequences=[X],
                                      outputs_info=[
                                          T.alloc(np.asarray(0, theano.config.floatX), sample_num, self.hidden_dim),
                                          T.alloc(np.asarray(0, theano.config.floatX), sample_num, self.hidden_dim)],
                                      n_steps=step_num)

        pass
