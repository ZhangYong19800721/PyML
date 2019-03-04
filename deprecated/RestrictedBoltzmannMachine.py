# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import optimal
import mydataset


class RestrictedBoltzmannMachine(object):
    """
        约束玻尔兹曼机（RBM）
    """

    def __init__(self):
        w = T.dmatrix('w')
        bv = T.dvector('bv')  # 显层的偏置
        bh = T.dvector('bh')  # 隐层的偏置

        v = T.dmatrix('v')  # 显层的状态
        h = T.dmatrix('h')  # 隐层的状态

        hp = T.nnet.sigmoid(v.dot(w) + bh)  # 隐层的激活概率计算公式
        vp = T.nnet.sigmoid(h.dot(w.T) + bv)  # 显层的激活概率计算公式

        self.f_foreward = theano.function([v, w, bv, bh], hp, on_unused_input='ignore')  # 前向传播函数
        self.f_backward = theano.function([h, w, bv, bh], vp, on_unused_input='ignore')  # 反向传播函数

        self.datas = None  # 开始训练之前需要绑定训练数据
        self.parameters = None  # 训练完毕之后需要绑定模型参数，参数顺序为[W,Bv,Bh]
        self.minibatch_size = 100  # 设定minibatch的大小

    def do_foreward(self, x):
        hp = self.f_foreward(x, *self.parameters)  # 计算隐层的激活概率
        return hp, (np.random.rand(*hp.shape) < hp) + 0.0  # 抽样

    def do_backward(self, x):
        vp = self.f_backward(x, *self.parameters)  # 计算显层的激活概率
        return vp, (np.random.rand(*vp.shape) < vp) + 0.0  # 抽样

    def do_gradient_vector(self, step_idx, *x):
        self.parameters = x  # 设定模型参数
        if self.minibatch_size > 0:
            N = self.datas.shape[0]  # 总训练样本数量
            minibatch_num = N // self.minibatch_size  # 得到minibatch的个数
            minibatch_idx = step_idx % minibatch_num  # 得到当前该取哪个minibatch
            select = list(range(minibatch_idx * self.minibatch_size, (1 + minibatch_idx) * self.minibatch_size))
            v0 = self.datas[select, :]  # 从训练数据中选择若干个样本组成一个minibatch
        else:
            v0 = self.datas

        h0p, h0 = self.do_foreward(v0)
        v1p, v1 = self.do_backward(h0)
        h1p, h1 = self.do_foreward(v1p)

        weight_cost = 1e-4
        gw = -(v0.T.dot(h0p) - v1p.T.dot(h1p)) / v0.shape[0] + weight_cost * self.parameters[0]
        gbh = -(h0 - h1p).sum(axis=0).T / v0.shape[0]
        gbv = -(v0 - v1p).sum(axis=0).T / v0.shape[0]
        return [gw, gbv, gbh]

    def do_object_function(self, step_idx, *x):
        self.parameters = x
        if self.minibatch_size > 0:
            N = self.datas.shape[0]
            minibatch_num = N // self.minibatch_size  # 得到minibatch的个数
            minibatch_idx = step_idx % minibatch_num  # 得到当前该取哪个minibatch
            select = list(range(minibatch_idx * self.minibatch_size, (1 + minibatch_idx) * self.minibatch_size))
            v0 = self.datas[select, :]  # 从训练数据中选择若干个样本组成一个minibatch
        else:
            v0 = self.datas

        h0p, h0 = self.do_foreward(v0)
        v1p, v1 = self.do_backward(h0)
        return (((v1p - v0) ** 2).sum(axis=1)).mean()


if __name__ == '__main__':
    # 准备训练数据
    train_datas, train_label, test_datas, test_label = mydataset.load_mnist_k()

    # 初始化模型参数
    W = 0.01 * np.random.randn(784, 2000)
    Bv = np.mean(train_datas, axis=0)
    Bv = np.log(Bv / (1 - Bv))
    Bv[Bv < -100] = -100
    Bv[Bv > +100] = +100
    Bh = np.zeros((2000,))

    model = RestrictedBoltzmannMachine()

    # 绑定训练数据
    model.datas = train_datas
    x_optimal, y_optimal = optimal.minimize_SGD(model, W, Bv, Bh, max_step=1000000, learn_rate=0.01, window=600)
    # print(x_optimal)
