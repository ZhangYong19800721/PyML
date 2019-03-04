# -*- coding: utf-8 -*-
"""
    带softmax神经元的约束玻尔兹曼机（SoftmaxRBM）
"""

import numpy as np
import theano
import theano.tensor as T
import collections
from optimizer import minimize
from dataset import mnist
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams()


class SoftmaxRBM(object):
    """
        带softmax神经元的约束玻尔兹曼机（SoftmaxRBM）
    """

    # 构造函数
    # 输入：
    #      softmax_dim (int) softmax 神经元的个数（分类类别）
    #      visual_dim (int) 可见神经元的个数
    #      hidden_dim (int) 隐藏神经元的个数
    def __init__(self, softmax_dim, visual_dim, hidden_dim):
        self.softmax_dim = softmax_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.parameters = collections.OrderedDict()  # 模型的参数集合

    # 初始化模型参数
    # 输入：
    #     Wsh （numpy ndarray，None）从softmax神经元到隐藏神经元的权值矩阵，默认为None
    #     Wvh （numpy ndarray，None）从visual神经元到隐藏神经元的权值矩阵，默认为None
    #     Bs（numpy ndarray，None）softmax神经元偏置，默认为None
    #     Bv（numpy ndarray，None）visual神经元偏置，默认为None
    #     Bh（numpy ndarray，None）hidden神经元偏置，默认为None
    def initialize_parameters(self, Wsh=None, Wvh=None, Bs=None, Bv=None, Bh=None):
        if Wsh is not None:
            softmax_dim = Wsh.shape[0]
            hidden_dim = Wsh.shape[1]
            assert softmax_dim == self.softmax_dim
            assert hidden_dim == self.hidden_dim
        else:
            Wsh = np.asarray(0.1 * np.random.randn(self.softmax_dim, self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Wsh'] = theano.shared(Wsh, name='Wsh')  # 从softmax神经元到hidden神经元的权值矩阵

        if Wvh is not None:
            visual_dim = Wvh.shape[0]
            hidden_dim = Wvh.shape[1]
            assert visual_dim == self.visual_dim
            assert hidden_dim == self.hidden_dim
        else:
            Wvh = np.asarray(0.1 * np.random.randn(self.visual_dim, self.hidden_dim), dtype=theano.config.floatX)
        self.parameters['Wvh'] = theano.shared(Wvh, name='Wvh')  # 从visual神经元到hidden神经元的权值矩阵

        if Bs is not None:
            softmax_dim = Bs.shape[0]
            assert softmax_dim == self.softmax_dim
        else:
            Bs = np.asarray(np.zeros(self.softmax_dim), dtype=theano.config.floatX)  # softmax 的偏置
        self.parameters['Bs'] = theano.shared(Bs, name='Bs')  # softmax 的偏置

        if Bv is not None:
            visual_dim = Bv.shape[0]
            assert visual_dim == self.visual_dim
        else:
            Bv = np.asarray(np.zeros(self.visual_dim), dtype=theano.config.floatX)  # 显层的偏置
        self.parameters['Bv'] = theano.shared(Bv, name='Bv')  # 显层的偏置

        if Bh is not None:
            hidden_dim = Bh.shape[0]
            assert hidden_dim == self.hidden_dim
        else:
            Bh = np.asarray(np.zeros(self.hidden_dim), dtype=theano.config.floatX)  # 隐层的偏置
        self.parameters['Bh'] = theano.shared(Bh, name='Bh')  # 隐层的偏置

    # 初始化模型
    def initialize_model(self):
        # 前向运算函数
        # S, softmax 的状态
        # V, visual  的状态
        # P, 模型参数
        def forward(S, V, **P):
            SV = T.concatenate([S, V], axis=1)
            WW = T.concatenate([P['Wsh'], P['Wvh']], axis=0)
            Hp = T.nnet.sigmoid(SV.dot(WW) + P['Bh'])  # hidden 激活概率计算公式
            Hs = (srng.uniform(Hp.shape) < Hp) + 0  # hidden  抽样状态。加0是为了把一个bool矩阵转换为整数矩阵
            return Hp, Hs

        # 反向运算函数
        # 输入：
        # H, hidden  的状态
        # P, 模型参数
        def backward(H, **P):
            Sp = T.nnet.softmax(H.dot(P['Wsh'].T) + P['Bs'])  # softmax 激活概率计算公式
            Vp = T.nnet.sigmoid(H.dot(P['Wvh'].T) + P['Bv'])  # visual  激活概率计算公式
            CC = T.extra_ops.cumsum(Sp, axis=1) < srng.uniform((Sp.shape[0], 1))
            Ss = T.basic.choose(T.reshape(CC.sum(axis=1), (-1, 1)), T.eye(Sp.shape[1]))  # softmax 抽样状态
            Vs = (srng.uniform(Vp.shape) < Vp) + 0  # visual  抽样状态。加0是为了把一个bool矩阵转换为整数矩阵
            return Sp, Vp, Ss, Vs

        S = T.fmatrix('S')  # softmax 的状态
        V = T.fmatrix('V')  # visual 的状态
        H = T.fmatrix('H')  # hidden 的状态

        Hp, Hs = forward(S, V, **self.parameters)
        Sp, Vp, Ss, Vs = backward(H, **self.parameters)

        self._f_foreward = theano.function([S, V], [Hp, Hs])  # 前向传播函数,输出隐层的激活概率和状态抽样
        self._f_backward = theano.function([H], [Sp, Vp, Ss, Vs])  # 反向传播函数,输出softmax & visual的激活概率和状态抽样

        X = T.fmatrix('X')  # 网络的输入
        Y = T.fmatrix('Y')  # 网络的输出(期望输出)
        V0s = X  # visual  层的初始状态
        S0s = Y  # softmax 层的初始状态
        H0p, H0s = forward(S0s, V0s, **self.parameters)
        S1p, V1p, S1s, V1s = backward(H0s, **self.parameters)
        H1p, H1s = forward(S1s, V1p, **self.parameters)

        cost = (((S1p - Y) ** 2).sum(axis=1) + ((V1p - X) ** 2).sum(axis=1)).mean()  # 整体重建误差
        sampleNum = T.cast(X.shape[0], 'floatX')
        weight_cost = 1e-4
        grad_Wsh = -(S0s.T.dot(H0p) - S1s.T.dot(H1p)) / sampleNum + weight_cost * self.parameters[
            'Wsh']  # Wsh的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Wvh = -(V0s.T.dot(H0p) - V1p.T.dot(H1p)) / sampleNum + weight_cost * self.parameters[
            'Wvh']  # Wvh的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Bs = -(S0s - S1p).mean(axis=0).T  # Bs的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Bv = -(V0s - V1p).mean(axis=0).T  # Bv的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）
        grad_Bh = -(H0s - H1p).mean(axis=0).T  # Bh的梯度（这并非真正的梯度，而是根据CD1算法得到近似梯度）

        self.grad = [theano.shared(p.get_value() * 0.0, name=f'grad_{k}') for k, p in self.parameters.items()]
        grad = [grad_Wsh, grad_Wvh, grad_Bs, grad_Bv, grad_Bh]  # 梯度
        updates = [(_g, g) for _g, g in zip(self.grad, grad)]
        self._f_grad = theano.function([X, Y], cost, updates=updates)  # 该函数计算模型的梯度，但是不更新参数

    # 前向过程
    def f_forward(self, S, V):
        return self._f_foreward(S, V)

    # 反向过程
    def f_backward(self, H):
        return self._f_backward(H)

    # 计算梯度,梯度向量被放在self.grad(theano shared variable)中，该函数会返回目标函数的值
    def f_grad(self, X, Y):
        return self._f_grad(X, Y)


if __name__ == '__main__':
    # 准备训练数据 TODO
    train_set = mnist.train_set('./data/mnist.mat')
    images, labels = train_set[0:len(train_set)]
    Bv = np.mean(images, axis=0)
    Bv = np.log(Bv / (1 - Bv))
    Bv[Bv < -100] = -100
    Bv[Bv > +100] = +100

    labels = np.choose(labels.astype('int'), np.eye(10, dtype='float32'))
    Bs = np.mean(labels, axis=0)
    Bs = np.log(Bs / (1 - Bs))
    Bs[Bs < -100] = -100
    Bs[Bs > +100] = +100

    train_set.set_minibatch_size(100)
    softmax_rbm = SoftmaxRBM(10, 784, 2000)
    softmax_rbm.initialize_parameters(Bs=Bs, Bv=Bv)
    softmax_rbm.initialize_model()

    optimizer = minimize.SGD(softmax_rbm)
    optimizer.train(train_set)
