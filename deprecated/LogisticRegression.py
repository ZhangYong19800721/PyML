# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal


class LogisticRegression(object):
    """
        逻辑回归器
    """

    def __init__(self):
        w = T.dmatrix('w')  # 权值
        b = T.dvector('b')  # 偏置
        x = T.dmatrix('x')  # 模型的输入
        y = T.dmatrix('y')  # 模型的输出
        n = T.dot(x, w) + b  # 净输出
        p = T.nnet.sigmoid(n)  # 概率值
        model_predict = p > 0.5  # 0.5为预测的门限值
        cross_entropy = -y * T.log(p) - (1 - y) * T.log(1 - p)  # 交叉熵
        object_function = cross_entropy.mean() + 1e-4 * (w ** 2).sum()
        gradient_vector = T.grad(object_function, [w, b])  # 梯度向量

        self.f_model_predict = theano.function([x, w, b], model_predict)
        self.f_object_function = theano.function([x, y, w, b], object_function)
        self.f_gradient_vector = theano.function([x, y, w, b], gradient_vector)

        self.datas = None  # 开始训练之前需要绑定训练数据
        self.label = None  # 开始训练之前需要绑定训练标签
        self.parameters = None  # 训练完毕之后需要绑定模型参数

    def do_model_predict(self, x):
        return self.f_model_predict(x, *self.parameters)

    def do_object_function(self, *x):
        return self.f_object_function(self.datas, self.label, *x)

    def do_gradient_vector(self, *x):
        return self.f_gradient_vector(self.datas, self.label, *x)


if __name__ == '__main__':
    model = LogisticRegression()

    N = 2000
    train_datas = np.random.rand(N, 2)
    train_label = np.zeros((N, 1))

    for n in range(N):
        if train_datas[n][0] > train_datas[n][1]:
            train_label[n] = 0
        else:
            train_label[n] = 1

    w = 0.01 * np.random.randn(2, 1)
    b = np.zeros((1,))

    model.datas = train_datas  # 开始训练之前需要绑定训练数据
    model.label = train_label  # 开始训练之前需要绑定训练标签
    z = model.do_object_function(w, b)
    g = model.do_gradient_vector(w, b)
    print(z)
    print(g)

    x_optimal, y_optimal = optimal.minimize_GD(model, w, b, max_step=100000, learn_rate=1e-3)

    model.parameters = x_optimal
    predict = model.do_model_predict(train_datas)

    x1, y1, x2, y2 = [], [], [], []
    for n in range(N):
        if predict[n]:
            x1.append(train_datas[n][0])
            y1.append(train_datas[n][1])
        else:
            x2.append(train_datas[n][0])
            y2.append(train_datas[n][1])

    plt.plot(x1, y1, '+r', x2, y2, '*g')
    plt.show()
