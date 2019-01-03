# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import optimal

class RestrictedBoltzmannMachine(object):
    """
        约束玻尔兹曼机（RBM）
    """
    def __init__(self): 
        w = T.dmatrix('w')
        bv = T.dvector('bv') # 显层的偏置
        bh = T.dvector('bh') # 隐层的偏置
            
        v = T.dmatrix('v') # 显层的状态
        h = T.dmatrix('h') # 隐层的状态

        hp = T.nnet.sigmoid(v.dot(w)+bh)   # 隐层的激活概率计算公式
        vp = T.nnet.sigmoid(h.dot(w.T)+bv) # 显层的激活概率计算公式
        
        self.f_foreward = theano.function([v,w,bv,bh],hp,on_unused_input='ignore') # 前向传播函数
        self.f_backward = theano.function([h,w,bv,bh],vp,on_unused_input='ignore') # 反向传播函数
        
        self.datas = None # 开始训练之前需要绑定训练数据
        self.parameters = None # 训练完毕之后需要绑定模型参数
           
    def do_foreward(self,x):
        hp = self.f_foreward(x,*self.parameters) # 计算隐层的激活概率
        return hp, (np.random.rand(*hp.shape) < hp) + 0.0 # 抽样
    
    def do_backward(self,x):
        vp = self.f_backward(x,*self.parameters) # 计算显层的激活概率
        return vp, (np.random.rand(*vp.shape) < vp) + 0.0 # 抽样
    
    def do_gradient_vector(self,*x):
        self.parameters = x
        v0 = self.datas
        h0p,h0 = self.do_foreward(v0)
        v1p,v1 = self.do_backward(h0)
        h1p,h1 = self.do_foreward(v1p)
        gw = -(v0.T.dot(h0p) - v1p.T.dot(h1p)) / v0.shape[0]
        gbh = -(h0p - h1p).sum(axis=0).T / v0.shape[0]
        gbv = -(v0 - v1p).sum(axis=0).T / v0.shape[0]
        return [gw,gbv,gbh]
    
    def do_object_function(self,*x):
        self.parameters = x
        v0 = self.datas
        h0p,h0 = self.do_foreward(v0)
        v1p,v1 = self.do_backward(h0)
        return (((v1p - v0)**2).sum(axis=0)).mean()
        
if __name__ == '__main__':
    # 准备训练数据
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'],dtype=float).T / 255
    train_label = np.zeros((mnist['mnist_train_labels'].shape[0],10))
    for n in range(mnist['mnist_train_labels'].shape[0]):
        train_label[n,mnist['mnist_train_labels'][n]] = 1
    test_datas  = np.array(mnist['mnist_test_images'],dtype=float).T / 255
    test_label  = np.zeros((mnist['mnist_test_labels'].shape[0],10))
    for n in range(mnist['mnist_test_labels'].shape[0]):
        test_label[n,mnist['mnist_test_labels'][n]] = 1
    
    # 初始化模型参数
    w = 0.01 * np.random.randn(784,500)
    bv = np.zeros((784,))
    bh = np.zeros((500,))
    
    model = RestrictedBoltzmannMachine()
    
    # 绑定训练数据
    model.datas = train_datas
    x_optimal,y_optimal = optimal.minimize_GD(model,w,bv,bh,max_step=1000,learn_rate=0.001)
    print(x_optimal)