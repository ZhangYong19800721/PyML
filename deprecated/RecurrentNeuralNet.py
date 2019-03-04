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
        Ht = T.dmatrix('Ht')  # 网络的隐藏输入（t时刻）

        # 网络的输入
        Xt = T.dmatrix('Xt')  # 网络的外部输入（t时刻）

        # 网络的输出
        Zt = T.nnet.sigmoid(Ht.dot(Whh) + Xt.dot(Wvh) + Bh)  # 隐藏神经元状态输出（t时刻）
        Yt = Zt.dot(Who) + Bo  # 输出层的输出（t时刻）
        Lt = T.dmatrix('Lt')  # 网络的期望输出（标签）（t时刻）

        object_function = 0.5 * (((Yt - Lt) ** 2).sum(axis=1)).mean()  # 目标函数
        gradient_vector = T.grad(object_function, [Wvh, Whh, Who, Bh, Bo, Ht])  # 目标函数对模型参数的梯度

        self.f_object_function = theano.function([Xt, Lt] + [Wvh, Whh, Who, Bh, Bo, Ht], object_function)
        self.f_foreward = theano.function([Xt] + [Wvh, Whh, Who, Bh, Bo, Ht], [Zt, Yt])
        self.f_backward = theano.function([Xt, Lt] + [Wvh, Whh, Who, Bh, Bo, Ht], gradient_vector)

        self.datas = None  # 开始训练之前需要绑定训练数据，对于SimpleRNN来说，训练数据是一个列表，
        # 第n个元素表示t=n时刻的输入。每个元素都是一个np.array，每一行表示一个数据样本
        self.label = None  # 开始训练之前需要绑定训练标签，对于SimpleRNN来说，训练标签是一个列表，
        # 第n个元素表示t=n时刻的标签。每个元素都是一个np.array，每一行表示一个数据样本
        self.parameters = None  # 训练完毕之后需要绑定模型参数，参数顺序为[Wvh, Whh, Who, Bh, Bo]
        self.minibatch_size = 100  # 设定minibatch的大小

    def do_model_predict(self, x):
        """
            根据模型进行预测，即根据输入序列x和初始隐藏状态h预测输出序列，注意使用该函数之前需要绑定模型参数
            输入：
                x 是一个列表，元素为np.array，表示输入序列
            输出：
                预测序列
        """
        Wvh, Whh, Who, Bh, Bo, H0 = self.parameters
        Ht = H0
        Y,H = [],[]  # 输出序列
        for t in range(len(x)):
            Ht, Yt = self.f_foreward(x[t], Wvh, Whh, Who, Bh, Bo, Ht)
            Y.append(Yt)
            H.append(Ht)
        return Y,H

    def do_object_function(self, step_idx, *x):
        """
            计算模型的目标函数
            输入：
                x 模型的所有参数组成的列表，需要特别注意参数的顺序
                step_idx 表示迭代次数
            输出：
                模型的目标函数值
        """
        self.parameters = x  # 绑定参数，参数绑定之后可以使用do_model_predict函数了
        sample_num = self.datas[0].shape[0]  # 得到训练样本的个数

        if self.minibatch_size > 0:
            minibatch_num = sample_num // self.minibatch_size  # 得到minibatch的个数
            minibatch_idx = step_idx % minibatch_num  # 得到当前该取哪个minibatch
            select = list(range(minibatch_idx * self.minibatch_size, (1 + minibatch_idx) * self.minibatch_size))
            minibatch_datas = [d[select, :] for d in self.datas]
            minibatch_label = [l[select, :] for l in self.label]
            Y = self.do_model_predict(minibatch_datas)  # 使用训练数据做模型预测
            F = sum([((Yt - Lt) ** 2).sum(axis=1) for Yt, Lt in zip(Y, minibatch_label)]).mean()  # 计算目标函数
        else:
            Y = self.do_model_predict(self.datas)  # 使用训练数据做模型预测
            F = sum([((Yt - Lt) ** 2).sum(axis=1) for Yt, Lt in zip(Y, self.label)]).mean()  # 计算目标函数
        return F

    def do_gradient_vector(self, step_idx, *x):
        """
            计算模型的梯度向量
            输入：
                x 模型的所有参数组成的列表，需要特别注意参数的顺序
                step_idx 表示迭代次数
            输出：
                模型的梯度向量，列表，按顺序对应所有参数的梯度
        """
        hidden_num = x[0].shape[1]  # 得到隐藏神经元的个数
        sample_num = self.datas[0].shape[0]  # 得到训练样本的个数
        h0 = np.zeros((sample_num, hidden_num))  # 隐藏神经元的初值
        t = len(self.datas) - 1
        while t >= 0:
            pass
        pass


if __name__ == '__main__':
    print('start')

    Wvh = np.random.randn(4, 8)
    Whh = np.random.randn(8, 8)
    Who = np.random.randn(8, 1)
    Bh = np.zeros((8,))
    Bo = np.zeros((1,))
    H0 = np.zeros((1,8))

    model = RecurrentNeuralNet()
    model.parameters = [Wvh, Whh, Who, Bh, Bo, H0]

    x = [np.ones((1, 4)), 2 * np.ones((1, 4)), 3 * np.ones((1, 4)), 4 * np.ones((1, 4))]
    Y,H = model.do_model_predict(x)
    print(f'Y={Y}')
    print(f'H={H}')
