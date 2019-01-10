# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np

def softmax_sample(prob):
    s = np.zeros(prob.shape) 
    for n in range(s.shape[0]):
        i = np.random.choice(s.shape[1],1,p=prob[n,:])
        s[n,i] = 1
    return s

def load_mnist():
    """加载mnist训练数据"""
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'],dtype=float).T / 255
    train_label = np.zeros((mnist['mnist_train_labels'].shape[0],10))
    for n in range(mnist['mnist_train_labels'].shape[0]):
        train_label[n,mnist['mnist_train_labels'][n]] = 1
    test_datas  = np.array(mnist['mnist_test_images'],dtype=float).T / 255
    test_label  = mnist['mnist_test_labels'].reshape(-1)
    return train_datas, train_label, test_datas, test_label

def load_mnist_g():
    """加载mnist训练数据，并分组为minibatch，每个minibatch包含100个样本"""
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'],dtype=float).T / 255
    train_label = np.array(mnist['mnist_train_labels']).reshape(-1)
    N = train_datas.shape[0]
    minibatch_size = 100
    for n in range(N):
        idx = n % 10
        if idx != train_label[n]:
            flag = False
            for k in range(n+1,N):
                if idx == train_label[k]:
                    swap = train_label[k]
                    train_label[k] = train_label[n]
                    train_label[n] = swap
                    
                    swap = np.array(train_datas[k,:])
                    train_datas[k,:] = train_datas[n,:]
                    train_datas[n,:] = swap
                    
                    flag = True
                    break
            
            if flag == False:
                minibatch_num = n // minibatch_size
                break
    
    train_datas = train_datas[0:(minibatch_num*minibatch_size),:]
    train_label = train_label[0:(minibatch_num*minibatch_size)]
    index_label = np.zeros((train_label.shape[0],10))
    for n in range(train_label.shape[0]):
        index_label[n,train_label[n]] = 1
    train_label = index_label
    test_datas  = np.array(mnist['mnist_test_images'],dtype=float).T / 255
    test_label  = mnist['mnist_test_labels'].reshape(-1)
    return train_datas, train_label, test_datas, test_label
    