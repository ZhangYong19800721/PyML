# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import optimal
import utility

class SoftmaxRestrictedBoltzmannMachine(object):
    """
        带softmax神经元的约束玻尔兹曼机（SoftmaxRBM）
    """
    def __init__(self): 
        """构造函数"""
        s = T.dmatrix('s') # softmax神经元的状态
        v = T.dmatrix('v') # 显层的状态
        h = T.dmatrix('h') # 隐层的状态
        Wsh = T.dmatrix('Wsh') # softmax神经元到隐层的权值
        Wvh = T.dmatrix('Wvh') # 显层神经元到隐层的权值
        bs = T.dvector('bs') # softmax神经元的偏置
        bv = T.dvector('bv') # 显层的偏置
        bh = T.dvector('bh') # 隐层的偏置
        
        hp = T.nnet.sigmoid(T.concatenate([s,v],axis=1).dot(T.concatenate([Wsh,Wvh],axis=0))+bh)   # 隐层的激活概率计算公式
        vp = T.nnet.sigmoid(h.dot(Wvh.T)+bv) # 显层的激活概率计算公式
        sp = T.nnet.softmax(h.dot(Wsh.T)+bs) # softmax层激活概率计算公式
        
        self.f_foreward = theano.function([s,v] + [Wsh,Wvh,bs,bv,bh],    hp ,on_unused_input='ignore') # 前向传播函数
        self.f_backward = theano.function([h]   + [Wsh,Wvh,bs,bv,bh],[sp,vp],on_unused_input='ignore') # 反向传播函数
        
        self.datas = None # 开始训练之前需要绑定训练数据
        self.label = None # 开始训练之前需要绑定训练标签(即softmax节点的状态)
        self.parameters = None # 训练完毕之后需要绑定模型参数，顺序为[Wsh,Wvh,Bs,Bv,Bh]
        self.minibatch_size = 100 # 设定minibatch的大小
           
    def do_foreward(self,s,v):
        """根据softmax+显层状态值计算隐层的激活概率并抽样得到隐层的状态值"""
        hp = self.f_foreward(s,v,*self.parameters) # 计算隐层的激活概率
        return hp, (np.random.rand(*hp.shape) < hp) + 0.0 # 抽样
    
    def do_backward(self,h):
        """根据隐层状态值计算softmax+显层的激活概率并抽样得到显层的状态值"""
        sp,vp = self.f_backward(h,*self.parameters) # 计算softmax+显层的激活概率
        s = utility.softmax_sample(sp) # 对sp进行抽样
        v = (np.random.rand(*vp.shape) < vp) + 0.0 # 对vp进行抽样
        return sp, s, vp, v 
    
    def do_gradient_vector(self,step_idx,*x):
        """使用CD1快速算法来估计梯度，没有真正地计算梯度"""
        self.parameters = x # 设定模型参数
        
        if self.minibatch_size > 0:
            N = self.datas.shape[0] # 总训练样本数量
            minibatch_num = N // self.minibatch_size # 得到minibatch的个数
            minibatch_idx = step_idx % minibatch_num # 得到当前该取哪个minibatch
            select = list(range(minibatch_idx * self.minibatch_size, (1 + minibatch_idx) * self.minibatch_size))
            s0 = self.label[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            v0 = self.datas[select,:] # 从训练数据中选择若干个样本组成一个minibatch
        else:
            s0 = self.label # 全部训练数据组成一个batch
            v0 = self.datas # 全部训练数据组成一个batch
            
        h0p,h0 = self.do_foreward(s0,v0)
        s1p,s1,v1p,v1 = self.do_backward(h0)
        h1p,h1 = self.do_foreward(s1,v1p)
        
        weight_cost = 1e-4
        gWsh = -(s0.T.dot(h0p) -  s1.T.dot(h1p)) / self.minibatch_size + weight_cost * self.parameters[0]
        gWvh = -(v0.T.dot(h0p) - v1p.T.dot(h1p)) / self.minibatch_size + weight_cost * self.parameters[1]
        gBs = -(s0 - s1p).mean(axis=0).T
        gBv = -(v0 - v1p).mean(axis=0).T
        gBh = -(h0 - h1p).mean(axis=0).T
        return [gWsh,gWvh,gBs,gBv,gBh]
        
    def do_object_function(self,step_idx,*x):
        """根据CD1算法，计算约束玻尔兹曼机的一阶重建误差"""
        self.parameters = x
        if self.minibatch_size > 0:
            N = self.datas.shape[0] # 总训练样本数量
            minibatch_num = N // self.minibatch_size # 得到minibatch的个数
            minibatch_idx = step_idx % minibatch_num # 得到当前该取哪个minibatch
            select = list(range(minibatch_idx * self.minibatch_size, (1 + minibatch_idx) * self.minibatch_size))
            # select = np.random.choice(N,self.minibatch_size,replace=False) # 按照minibatch的大小产生若干个随机数,不会重复选中
            s0 = self.label[select,:] # 从训练数据中选择若干个样本组成一个minibatch
            v0 = self.datas[select,:] # 从训练数据中选择若干个样本组成一个minibatch
        else:
            s0 = self.label # 全部训练数据组成一个batch
            v0 = self.datas # 全部训练数据组成一个batch
            
        h0p,h0 = self.do_foreward(s0,v0)
        s1p,s1,v1p,v1 = self.do_backward(h0)
        return (((s1p - s0)**2).sum(axis=1)).mean() + (((v1p - v0)**2).sum(axis=1)).mean()
    
    def do_model_predict(self,x):
        """使用模型进行分类预测"""
        Wsh,Wvh,Bs,Bv,Bh = self.parameters # 得到模型的参数
        num_samples = x.shape[0] # 需要预测的样本数量
        num_softmax = Bs.shape[0] # 类别数量
        free_energy = sys.float_info.max * np.ones((num_samples,num_softmax)) # 初始化自由能量为最大值
        
        for n in range(num_softmax):
            s = np.zeros((num_samples,num_softmax))
            s[:,n] = 1
            free_energy[:,n] = - s.dot(Bs) - x.dot(Bv) - np.sum(np.log(1 + np.exp(s.dot(Wsh) + x.dot(Wvh) + Bh)),axis=1)
    
        return np.argmin(free_energy,axis=1)
        
if __name__ == '__main__':
    # 准备训练数据
    train_datas,train_label,test_datas,test_label = utility.load_mnist()
    
    # 初始化模型参数
    Wsh = 0.01 * np.random.randn(10,2000)
    Wvh = 0.01 * np.random.randn(784,2000)
    Bs = np.mean(train_label,axis=0)
    Bs = np.log(Bs / (1-Bs))
    Bs[Bs<-100] = -100
    Bs[Bs>+100] = +100
    Bv = np.mean(train_datas,axis=0)
    Bv = np.log(Bv / (1-Bv))
    Bv[Bv<-100] = -100
    Bv[Bv>+100] = +100
    Bh = np.zeros((2000,))
    
    model = SoftmaxRestrictedBoltzmannMachine()
    
    # 绑定训练数据
    model.datas = train_datas
    model.label = train_label
    
    # 训练
    x_optimal,y_optimal = optimal.minimize_SGD(model,Wsh,Wvh,Bs,Bv,Bh,max_step=1000000,learn_rate=0.01,window=600)
    
    model.parameters = x_optimal # 绑定参数
    predict = model.do_model_predict(test_datas)
    
    error_rate = np.sum((predict != test_label) + 0.0) / len(test_label)
    print(f'error_rate = {error_rate}')
    
    