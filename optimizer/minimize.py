# -*- coding: utf-8 -*-
"""
    随机梯度下降SGD（Stochastic Gradient Decend）
"""

import numpy as np
import theano
import time


class SGD(object):
    def __init__(self, model, **options):
        self.model = model
        self.options = options
        if 'learn_rate' not in self.options:
            self.options['learn_rate'] = 0.01  # 缺省的学习速度

        if 'max_epoch' not in self.options:
            self.options['max_epoch'] = 100  # 缺省最多遍历全部的训练数据max_epoch遍

        if 'momentum' not in self.options:
            self.options['momentum'] = 0.9  # 缺省的动量参数

        self.delta = [theano.shared(p.get_value() * 0.0, name=f'delta_{k}') for k, p in self.model.parameters.items()]
        update_delta = [(d, self.options['momentum'] * d - self.options['learn_rate'] * g) for d, g in zip(self.delta, model.grad)]
        update_param = [(p, p + d) for p, d in zip(model.parameters.values(), self.delta)]
        self.f_update = theano.function([], [], updates=update_delta + update_param)

    def update(self, X, Y):
        cost = self.model.f_grad(X, Y)  # 计算目标函数和梯度，梯度被存储在model.grad中
        self.f_update()  # 更新梯度
        return cost

    def train(self, dataset):
        print("开始训练模型.....")
        begin = time.clock()  # 记录起始时间

        window = len(dataset)
        avcost = self.update(dataset[0][0], dataset[0][0])
        for epoch in range(self.options['max_epoch']):
            for minibatch_idx in range(len(dataset)):
                minibach, _ = dataset[minibatch_idx]
                cost = self.update(minibach, minibach)
                avcost = (1 - 1 / window) * avcost + (1 / window) * cost  # 更新滑动均值
                print('epoch: %3d, step: %5d, cost: %12.8f, average: %12.8f' % (epoch, minibatch_idx, cost, avcost))

        finish = time.clock()  # 记录结束时间
        print(f"训练结束，耗时{(finish - begin) / 60}分钟")
