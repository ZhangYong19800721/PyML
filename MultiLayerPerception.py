# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal

class MultiLayerPerception(object):
    """
        多层感知器
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
            
        model_predict = n[-1] # 模型的预测
        self.f_model_predict = theano.function([x]+w+b,model_predict)
        
        object_function = 0.5 * (((model_predict - y)**2).sum(axis=1)).mean() + 1e-4 * sum([(w_i**2).sum() for w_i in w])
        self.f_object_function = theano.function([x,y]+w+b,object_function)
        
        gradient_vector = T.grad(object_function,w+b)
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
    N = 2000
    train_datas = np.linspace(-np.pi,np.pi,N).reshape(N,1)
    train_label = np.sin(train_datas).reshape(N,1)
    
    model = MultiLayerPerception(3)
    model.datas = train_datas
    model.label = train_label
    
    w1 = 0.01*np.random.randn(1,8)
    b1 = np.zeros((8,))
    w2 = 0.01*np.random.randn(8,8)
    b2 = np.zeros((8,))
    w3 = 0.01*np.random.randn(8,1)
    b3 = np.zeros((1,))
    z = model.do_object_function(w1,w2,w3,b1,b2,b3)
    g = model.do_gradient_vector(w1,w2,w3,b1,b2,b3)
    print(z)
    print(g)
    
    x_optimal,y_optimal = optimal.minimize_GD(model,w1,w2,w3,b1,b2,b3,max_step=50000,learn_rate=1.0)
    
    model.parameters = x_optimal
    predict = model.do_model_predict(train_datas)
    plt.plot(train_datas,train_label,'r',train_datas,predict,'g')