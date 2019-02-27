# -*- coding: utf-8 -*-
"""
    随机梯度下降SGD（Stochastic Gradient Decend）
"""

import theano
import time

class SGD(object):
    def __init__(self, model, **options):
        self.model = model
        self.options = options
        if 'learn_rate' not in self.options:
            self.options['learn_rate'] = 0.01  # 缺省的学习速度

        if 'max_epoch' not in self.options:
            self.options['max_epoch'] = 30  # 缺省最多遍历全部的训练数据max_epoch遍

        if 'momentum' not in self.options:
            self.options['momentum'] = 0.9  # 缺省的动量参数

        updates = [(p, p - self.options['learn_rate'] * g) for p, g in zip(model.parameters.values(), model.grad)]
        self.f_update = theano.function([], [], updates=updates)

    def update(self, X, Y):
        cost = self.model.f_grad(X, Y)  # 计算目标函数和梯度，梯度被存储在model.grad中
        self.f_update()  # 更新梯度
        return cost
    
    def train(self,dataset):
        print("开始训练模型.....")
        begin = time.clock() # 记录起始时间
        
        for epoch in range(self.options['max_epoch']):
            for minibatch_idx in range(len(dataset)):
                minibach,_ = dataset[minibatch_idx]
                cost = self.update(minibach,minibach)
                print(f'epoch: {epoch}, step: {minibatch_idx}, cost: {cost}')
                
        finish = time.clock() # 记录起始时间
        print(f"训练结束，耗时{(finish - begin)/60}分钟")