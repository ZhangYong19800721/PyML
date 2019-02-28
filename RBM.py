# -*- coding: utf-8 -*-
"""
    约束玻尔兹曼机（RBM）
    基于theano实现，使用GPU训练
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import collections
from optimizer import minimize
from dataset import mnist
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams()


class RBM(object):
    """
        约束玻尔兹曼机（RBM）
    """

    # 构造函数
    # 输入：
    #      visualDim (int) 可见神经元的个数
    #      hiddenDim (int) 隐藏神经元的个数
    def __init__(self, visualDim, hiddenDim):
        self.visualDim = visualDim
        self.hiddenDim = hiddenDim
        self.parameters = collections.OrderedDict()  # 模型的参数集合

    # 初始化模型参数
    # 输入：
    #     W（numpy ndarray，None）权值矩阵，默认为None
    #     Bv（numpy ndarray，None）显层偏置，默认为None
    #     Bh（numpy ndarray，None）隐层偏置，默认为None
    def initialize_parameters(self, W=None, Bv=None, Bh=None):
        if W is not None:
            self.visualDim = W.shape[0]
            self.hiddenDim = W.shape[1]
        else:
            W = np.asarray(0.1 * np.random.randn(self.visualDim, self.hiddenDim), dtype=theano.config.floatX)  # 权值矩阵
        self.parameters['W'] = theano.shared(W, name='W')  # 权值矩阵

        if Bv is not None:
            visualDim = Bv.shape[0]
            assert visualDim == self.visualDim
        else:
            Bv = np.asarray(np.zeros(self.visualDim), dtype=theano.config.floatX)  # 显层的偏置
        self.parameters['Bv'] = theano.shared(Bv, name='Bv')  # 显层的偏置

        if Bh is not None:
            hiddenDim = Bh.shape[0]
            assert hiddenDim == self.hiddenDim
        else:
            Bh = np.asarray(np.zeros(self.hiddenDim), dtype=theano.config.floatX)  # 隐层的偏置
        self.parameters['Bh'] = theano.shared(Bh, name='Bh')  # 隐层的偏置

    # 初始化模型
    def initialize_model(self):
        V = T.fmatrix('V')  # 显层的状态
        H = T.fmatrix('H')  # 隐层的状态

        Hp = T.nnet.sigmoid(V.dot(self.parameters['W']) + self.parameters['Bh'])  # 隐层的激活概率计算公式
        Hs = (srng.uniform(Hp.shape) < Hp) + 0  # 隐层抽样状态。加0是为了把一个bool矩阵转换为整数矩阵
        Vp = T.nnet.sigmoid(H.dot(self.parameters['W'].T) + self.parameters['Bv'])  # 显层的激活概率计算公式
        Vs = (srng.uniform(Vp.shape) < Vp) + 0  # 显层抽样状态。加0是为了把一个bool矩阵转换为整数矩阵
        self.f_foreward = theano.function([V], [Hp, Hs])  # 前向传播函数,输出隐层的激活概率和状态抽样
        self.f_backward = theano.function([H], [Vp, Vs])  # 反向传播函数,输出显层的激活概率和状态抽样

        X = T.fmatrix('X')  # 网络的输入
        Y = T.fmatrix('Y')  # 网络的输出(期望输出)
        V0s = X  # 显层的初始状态
        H0p = T.nnet.sigmoid(V0s.dot(self.parameters['W']) + self.parameters['Bh'])  # 隐层的激活概率
        H0s = (srng.uniform(H0p.shape) < H0p) + 0  # 隐层的抽样状态
        V1p = T.nnet.sigmoid(H0s.dot(self.parameters['W'].T) + self.parameters['Bv'])  # 显层的重建激活概率
        H1p = T.nnet.sigmoid(V1p.dot(self.parameters['W']) + self.parameters['Bh'])  # 隐层的激活概率（这里使用V1p而不是V1s得到更好的无偏抽样）
        cost = ((V1p - Y) ** 2).sum(axis=1).mean()  # 整体重建误差
        sampleNum = T.cast(V0s.shape[0], 'floatX')
        weight_cost = 1e-4
        grad_W = -(V0s.T.dot(H0p) - V1p.T.dot(H1p)) / sampleNum + weight_cost * self.parameters['W']  # W的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Bv = -(V0s - V1p).sum(axis=0).T / sampleNum  # Bv的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Bh = -(H0s - H1p).sum(axis=0).T / sampleNum  # Bh的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）

        self.grad = [theano.shared(p.get_value() * 0.0, name=f'grad_{k}') for k, p in self.parameters.items()]
        grad = [grad_W, grad_Bv, grad_Bh]  # 梯度
        updates=[(_g,g) for _g,g in zip(self.grad, grad)]
        self.f_grad = theano.function([X,Y], cost, updates=updates)  # 该函数计算模型的梯度，但是不更新梯度

    # 前向过程
    def forward(self, V):
        return self.f_foreward(V)

    # 反向过程
    def backward(self, H):
        return self.f_backward(H)


if __name__ == '__main__':
    # 准备训练数据
    train_set = mnist.train_set('./data/mnist.mat')
    allimages = train_set[0:len(train_set)][0]
    Bv = np.mean(allimages, axis=0)
    Bv = np.log(Bv / (1 - Bv))
    Bv[Bv < -100] = -100
    Bv[Bv > +100] = +100
    train_set.set_minibatch_size(100)
    rbm = RBM(784, 2000)
    rbm.initialize_parameters(Bv=Bv)
    rbm.initialize_model()
    
    optimizer = minimize.SGD(rbm)
    optimizer.train(train_set)
