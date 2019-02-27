# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

def load_mnist_k():
    """加载mnist训练数据"""
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'], dtype=float).T / 255
    train_label = np.zeros((mnist['mnist_train_labels'].shape[0], 10))
    for n in range(mnist['mnist_train_labels'].shape[0]):
        train_label[n, mnist['mnist_train_labels'][n]] = 1
    test_datas = np.array(mnist['mnist_test_images'], dtype=float).T / 255
    test_label = mnist['mnist_test_labels'].reshape(-1)
    return train_datas, train_label, test_datas, test_label

def load_mnist():
    """加载mnist训练数据"""
    mnist = sio.loadmat('./data/mnist.mat')
    train_image = np.array(mnist['mnist_train_images']).reshape((28,28,1,-1)).transpose((3,2,0,1))
    train_label = np.array(mnist['mnist_train_labels'], dtype=np.long).reshape(-1)
    test_image = np.array(mnist['mnist_test_images']).reshape((28,28,1,-1)).transpose((3,2,0,1))
    test_label = np.array(mnist['mnist_test_labels'], dtype=np.long).reshape(-1)
    return train_image, train_label, test_image, test_label


def load_mnist_g():
    """加载mnist训练数据，并分组为minibatch，每个minibatch包含100个样本"""
    mnist = sio.loadmat('./data/mnist.mat')
    train_datas = np.array(mnist['mnist_train_images'], dtype=float).T / 255
    train_label = np.array(mnist['mnist_train_labels']).reshape(-1)
    N = train_datas.shape[0]
    minibatch_size = 100
    for n in range(N):
        idx = n % 10
        if idx != train_label[n]:
            flag = False
            for k in range(n + 1, N):
                if idx == train_label[k]:
                    swap = train_label[k]
                    train_label[k] = train_label[n]
                    train_label[n] = swap

                    swap = np.array(train_datas[k, :])
                    train_datas[k, :] = train_datas[n, :]
                    train_datas[n, :] = swap

                    flag = True
                    break

            if flag == False:
                minibatch_num = n // minibatch_size
                break

    train_datas = train_datas[0:(minibatch_num * minibatch_size), :]
    train_label = train_label[0:(minibatch_num * minibatch_size)]
    index_label = np.zeros((train_label.shape[0], 10))
    for n in range(train_label.shape[0]):
        index_label[n, train_label[n]] = 1
    train_label = index_label
    test_datas = np.array(mnist['mnist_test_images'], dtype=float).T / 255
    test_label = mnist['mnist_test_labels'].reshape(-1)
    return train_datas, train_label, test_datas, test_label

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10():
    train_image = np.zeros((0,3,32,32),dtype=np.uint8)
    train_label = np.zeros(0,dtype=np.long)
    for n in range(1,6):
        file = f"./data/cifar-10-python/cifar-10-batches-py/data_batch_{n}" # 文件名
        data = unpickle(file) # 从文件中读取数据
        batch_image = data[b'data']
        batch_image = batch_image.reshape((10000,32,32,3),order='F') # 训练集图像数据
        batch_image = batch_image.transpose((0,3,2,1))
        train_image = np.vstack((train_image,batch_image))
        batch_label = np.array(data[b'labels'],dtype=np.long)
        train_label = np.hstack((train_label,batch_label))
    
    file = f"./data/cifar-10-python/cifar-10-batches-py/test_batch" # 文件名
    data = unpickle(file) # 从文件中读取数据
    test_image = data[b'data']
    test_image = test_image.reshape((10000,32,32,3),order='F') # 测试集图像数据
    test_image = test_image.transpose((0,3,2,1))
    test_label = np.array(data[b'labels'],dtype=np.long) # 测试集标签数据
    return train_image,train_label,test_image,test_label