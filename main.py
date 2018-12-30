# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import minimize

w1 = theano.shared(0.001*np.random.randn(1,8),'w1')
w2 = theano.shared(0.001*np.random.randn(8,1),'w2')
b1 = theano.shared(np.zeros((8,)),'b1')
b2 = theano.shared(np.zeros((1,)),'b2')
weight = [w1,b1,w2,b2]
datas = T.dmatrix('datas') # 训练数据
label = T.dmatrix('label') # 训练标签

n1 = T.dot(datas,w1)+b1
y1 = 1/(1+T.exp(-n1))
n2 = T.dot(y1,w2)+b2
model = n2 # 网络的输出
f_model = theano.function([datas],model)
cost = 0.5 * ((model - label)**2).sum() + 1e-5 * ((w1**2).sum() + (w2**2).sum())
f_cost = theano.function([datas,label],cost)

N = 2000
train_datas = np.linspace(-np.pi,np.pi,N).reshape(N,1)
train_label = np.sin(train_datas).reshape(N,1)

minimize.GradientDescend(cost,datas,label,weight,train_datas,train_label,max_step=1000000)

predict = f_model(train_datas)
plt.plot(train_datas,train_label,'r',train_datas,predict,'g')