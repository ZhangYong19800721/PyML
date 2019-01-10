# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import optimal

class SoftmaxMultilayerPerception(object):
    """
        带softmax层的多层感知器分类器（SoftmaxMLP）
    """
    def __init__(self, L=2): 
        # L表示层数
        w,b = [],[] 
        for i in range(L):
            w.append(T.dmatrix(f'w{i}'))
            b.append(T.dvector(f'b{i}'))
            
        x = T.dmatrix('x') # 模型的输入
        y = T.dmatrix('y') # 模型的输出

        n = [T.dot(x,w[0])+b[0]] # 第0层的净输出
        z = []  # 每一层的限幅输出
        
        for i in range(1,L): # 当L=1时该循环不执行
            z.append(T.nnet.sigmoid(n[i-1])) # sigmoid函数
            n.append(T.dot(z[i-1],w[i])+b[i])
        
        # 网络的最末端是一个softmax分类器
        p = T.nnet.softmax(n[-1])  # 概率输出
        model_predict = p # 模型的预测
        object_function = -((y*T.log(p)).sum(axis=1)).mean() + 1e-4 * ((w[-1]**2).sum() + (b[-1]**2).sum())  
        gradient_vector = T.grad(object_function,w+b)
            
        self.f_model_predict = theano.function([x]+w+b,model_predict)
        self.f_object_function = theano.function([x,y]+w+b,object_function)
        self.f_gradient_vector = theano.function([x,y]+w+b,gradient_vector)
        
        self.datas = None # 开始训练之前需要绑定训练数据
        self.label = None # 开始训练之前需要绑定训练标签
        self.parameters = None # 训练完毕之后需要绑定模型参数，参数顺序为[w0,w1,...,b0,b1,...]
        self.minibatch_size = 100 # 设定minibatch的大小
        
    def do_model_predict(self,x):
        return self.f_model_predict(x,*self.parameters)
    
    def do_object_function(self,*x):
        if self.minibatch_size > 0:
            N = self.datas.shape[0] # 总训练样本数量
            select = np.random.choice(N,self.minibatch_size,replace=False) # 按照minibatch的大小产生若干个随机数,不会重复选中
            label_minibatch = self.label[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            datas_minibatch = self.datas[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            return self.f_object_function(datas_minibatch,label_minibatch,*x)
        else:
            return self.f_object_function(self.datas,self.label,*x)
    
    def do_gradient_vector(self,*x):
        if self.minibatch_size > 0:
            N = self.datas.shape[0] # 总训练样本数量
            select = np.random.choice(N,self.minibatch_size,replace=False) # 按照minibatch的大小产生若干个随机数,不会重复选中
            label_minibatch = self.label[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            datas_minibatch = self.datas[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            return self.f_gradient_vector(datas_minibatch,label_minibatch,*x)
        else:
            return self.f_gradient_vector(self.datas,self.label,*x)
    
if __name__ == '__main__':
    # 准备训练数据
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'],dtype=float).T / 255
    train_label = np.zeros((mnist['mnist_train_labels'].shape[0],10))
    for n in range(mnist['mnist_train_labels'].shape[0]):
        train_label[n,mnist['mnist_train_labels'][n]] = 1
    test_datas  = np.array(mnist['mnist_test_images'],dtype=float).T / 255
    test_label  = mnist['mnist_test_labels'].reshape(-1)
    
    # 模型参数
    model = SoftmaxMultilayerPerception()
    w1 = 0.01 * np.random.randn(784,2000)
    b1 = np.zeros((2000,))
    w2 = 0.01 * np.random.randn(2000,10)
    b2 = np.zeros((10,))
    
    # 训练
    model.datas = train_datas
    model.label = train_label
    x_optimal, y_optimal = optimal.minimize_SGD(model,w1,w2,b1,b2,max_step=100000,learn_rate=1e-2)
    
    # 绑定模型参数
    model.parameters = x_optimal # 绑定模型参数
    
    # 测试模型性能
    predict = model.do_model_predict(test_datas)
    predict = np.argmax(predict,axis = 1)
    error_rate = np.sum((predict != test_label) + 0.0) / len(test_label)
    print(f'error_rate = {error_rate}')