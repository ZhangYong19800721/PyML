# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import optimal
import utility


class RecurrentNeuralNet(object):
    """
        简单循环神经网络(SimpleRNN)
    """

    def __init__(self):
        # 网络的参数
        Wvh = T.dmatrix('Wvh')  # 从输入层到隐藏层的权值矩阵
        Whh = T.dmatrix('Whh')  # 从隐藏层到隐藏层的权值矩阵
        Who = T.dmatrix('Who')  # 从隐藏层到输出层的权值矩阵
        Bh = T.dvector('Bh')  # 隐藏层的偏置值
        Bo = T.dvector('Bo')  # 输出层的偏置值

        # 网络的输入
        X = T.dmatrix('X')  # 网络的外部输入
        H = T.dmatrix('H')  # 网络的状态输入

        # 网络的输出
        Z = T.nnet.sigmoid(H.dot(Whh) + X.dot(Wvh) + Bh)  # 隐藏神经元状态输出
        Y = Z.dot(Who) + Bo  # 输出层的输出
        L = T.dmatrix('L')  # 网络的期望输出（标签）

        object_function = 0.5 * (((Y - L) ** 2).sum(axis=1)).mean() # 目标函数
        gradient_vector = T.grad(object_function, [H, Wvh, Whh, Who, Bh, Bo]) # 梯度向量

        self.f_object_function = theano.function([H, X, L] + [Wvh, Whh, Who, Bh, Bo], object_function)
        self.f_foreward = theano.function([H, X] + [Wvh, Whh, Who, Bh, Bo], [Z, Y])

        self.datas = None  # 开始训练之前需要绑定训练数据
        self.label = None  # 开始训练之前需要绑定训练标签
        self.parameters = None  # 训练完毕之后需要绑定模型参数，参数顺序为[Wvh, Whh, Who, Bh, Bo]
        self.minibatch_size = 100  # 设定minibatch的大小

    def do_model_predict(self, h, x):  # x是一个列表，元素为np.array
        Y = []
        for n in range(len(x)):
            h, Yn = self.f_foreward(h, x[n], *self.parameters)
            Y.append(Yn)
        return Y


if __name__ == '__main__':
    print('start')

    Wvh = np.random.randn(4, 8)
    Whh = np.random.randn(8, 8)
    Who = np.random.randn(8, 1)
    bh = np.zeros((8,))
    bo = np.zeros((1,))

    model = RecurrentNeuralNet()
    model.parameters = [Wvh, Whh, Who, bh, bo]

    h = np.zeros((1, 8))
    x = [np.ones((1, 4)), 2 * np.ones((1, 4)), 3 * np.ones((1, 4)), 4 * np.ones((1, 4))]
    Y = model.do_model_predict(h, x)
    print(Y)
