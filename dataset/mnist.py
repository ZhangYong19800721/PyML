# -*- coding: utf-8 -*-
"""
    MNIST手写数字图片数据集
"""

import scipy.io as sio
import numpy as np


class train_set(object):
    def __init__(self, filepath):
        mnist = sio.loadmat(filepath)
        self.datas = np.array(mnist['mnist_train_images'], dtype='float32').T / 255.0
        self.label = np.array(mnist['mnist_train_labels'], dtype='int32')
        self.label = np.choose(self.label, np.eye(10, dtype='float32'))
        self.minibatch_size = 1

    def __len__(self):
        return self.datas.shape[0] // self.minibatch_size

    def __getitem__(self, item):
        if self.minibatch_size > 1:
            return self.datas[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :], \
                   self.label[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :]
        else:
            return self.datas[item], self.label[item]

    def set_minibatch_size(self, size):
        self.minibatch_size = size


class test_set(object):
    def __init__(self, filepath):
        mnist = sio.loadmat(filepath)
        self.datas = np.array(mnist['mnist_test_images'], dtype='float32').T / 255.0
        self.label = np.array(mnist['mnist_test_labels'], dtype='int32')
        self.minibatch_size = 1
        # self.label = np.choose(self.label, np.eye(10, dtype='float32'))

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, item):
        if self.minibatch_size > 1:
            return self.datas[(item * self.minibatch_size):((item + 1) * self.minibatch_size), :], \
                   self.label[(item * self.minibatch_size):((item + 1) * self.minibatch_size)]
        else:
            return self.datas[item], self.label[item]

    def set_minibatch_size(self, size):
        self.minibatch_size = size


if __name__ == '__main__':
    mnist_train_set = train_set('../data/mnist.mat')
    mnist_train_set.set_minibatch_size(100)
    train_images, train_lables = mnist_train_set[0]

    mnist_test_set = test_set('../data/mnist.mat')
    test_images, test_lables = mnist_test_set[0:len(mnist_test_set)]
