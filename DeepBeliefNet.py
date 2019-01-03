# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal

class DeepBeliefNet(object):
    """
        深度信度网络（DBN）
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
        object_function = -(y*T.log(p)).sum() + 1e-5 * (sum([(w_i**2).sum() for w_i in w]) + (b[-1]**2).sum()) 
        gradient_vector = T.grad(object_function,w+b)
            
        self.f_model_predict = theano.function([x]+w+b,model_predict)
        self.f_object_function = theano.function([x,y]+w+b,object_function)
        self.f_gradient_vector = theano.function([x,y]+w+b,gradient_vector)
        
        self.datas = None # 开始训练之前需要绑定训练数据
        self.label = None # 开始训练之前需要绑定训练标签
        self.parameters = None # 训练完毕之后需要绑定模型参数
        
    def do_model_predict(self,x):
        return self.f_model_predict(x,*self.parameters)
    
    def do_object_function(self,*x):
        return self.f_object_function(self.datas,self.label,*x)
    
    def do_gradient_vector(self,*x):
        return self.f_gradient_vector(self.datas,self.label,*x)
    
if __name__ == '__main__':
    # 准备训练数据
    
    # 模型参数
    model = DeepBeliefNet(4)
    w1 = 0.01 * np.random.randn(784,500)
    b1 = np.zeros((500,))
    w2 = 0.01 * np.random.randn(500,500)
    b2 = np.zeros((500,))
    w3 = 0.01 * np.random.randn(500,2000)
    b3 = np.zeros((2000,))
    w4 = 0.01 * np.random.randn(2000,10)
    b4 = np.zeros((10,))
    
    model.parameters = [w1,w2,w3,w4,b1,b2,b3,b4] # 绑定模型参数
    x = np.ones((3,784))
    predict = model.do_model_predict(x)
    
    #plt.plot(train_datas,train_label,'r',train_datas,predict,'g')