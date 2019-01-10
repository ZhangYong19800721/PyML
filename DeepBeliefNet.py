# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import optimal
import RestrictedBoltzmannMachine as RBM
import SoftmaxRestrictedBoltzmannMachine as SRBM
import utility

class DeepBeliefNet(object):
    """
        深度信度网络（DBN）
    """
    def __init__(self, L=2): 
        # L表示层数
        self.stack_RBM = [] # 栈式约束玻尔兹曼机
        for i in range(L-1):
            self.stack_RBM.append(RBM.RestrictedBoltzmannMachine())
        
        # 网络的最末端是一个softmaxRBM分类器
        self.softmax_RBM = SRBM.SoftmaxRestrictedBoltzmannMachine()
        
        self.datas = None # 开始训练之前需要绑定训练数据
        self.label = None # 开始训练之前需要绑定训练标签
        
    def bind_parameters(self,*x): 
        """绑定模型参数"""
        index = 0
        for rbm in self.stack_RBM:
            rbm.parameters = x[index:(index+3)]
            index = index + 3
        
        self.softmax_RBM.parameters = x[index:]
        
    def train(self, **options):
        # 对每一层RBM进行快速训练
        datas = self.datas # 初始化训练数据
        for rbm in self.stack_RBM:
            rbm.datas = datas # 绑定训练数据

            # 初始化参数
            W,Bv,Bh = rbm.parameters
            Bv = np.mean(rbm.datas,axis=0)
            Bv = np.log(Bv / (1-Bv))
            Bv[Bv<-100] = -100
            Bv[Bv>+100] = +100
            
            x_optimal,y_optimal = optimal.minimize_SGD(rbm,W,Bv,Bh,**options) # 训练
            rbm.parameters = x_optimal # 绑定最优参数
            datas, _1 = rbm.do_foreward(datas) # 映射训练数据到下一层
            
        self.softmax_RBM.datas = datas # 绑定训练数据
        self.softmax_RBM.label = self.label # 绑定训练标签
        x_optimal,y_optimal = optimal.minimize_SGD(self.softmax_RBM,*self.softmax_RBM.parameters,**options) # 训练
        self.softmax_RBM.parameters = x_optimal # 绑定最优参数
        
    def do_model_predict(self,x):
        z = x
        for rbm in self.stack_RBM:
            z,_1 = rbm.do_foreward(z)
        
        return self.softmax_RBM.do_model_predict(z)
        
if __name__ == '__main__':
    # 准备训练数据
    train_datas,train_label,test_datas,test_label = utility.load_mnist()
    
    # 创建模型
    model = DeepBeliefNet(3)
    
    # 第1层约束玻尔兹曼机参数
    W1 = 0.01 * np.random.randn(784,500)
    Bv1 = np.zeros((784,))
    Bh1 = np.zeros((500,))
    
    # 第2层约束玻尔兹曼机参数
    W2 = 0.01 * np.random.randn(500,500)
    Bv2 = np.zeros((500,))
    Bh2 = np.zeros((500,))
    
    # 第3层Softmax约束玻尔兹曼机参数
    Wsh = 0.01 * np.random.randn(10,2000)
    Wvh = 0.01 * np.random.randn(500,2000)
    Bs = np.zeros((10,))
    Bv = np.zeros((500,))
    Bh = np.zeros((2000,))
    
    # 绑定参数
    model.bind_parameters(W1,Bv1,Bh1,W2,Bv2,Bh2,Wsh,Wvh,Bs,Bv,Bh)
    
    # 绑定训练数据
    model.datas = train_datas
    model.label = train_label
    
    # 训练
    model.train(max_step=1000000,learn_rate=0.01,window=600)
        
    # 测试模型性能
    predict = model.do_model_predict(test_datas)
    
    error_rate = np.sum((predict != test_label) + 0.0) / len(test_label)
    print(f'error_rate = {error_rate}')