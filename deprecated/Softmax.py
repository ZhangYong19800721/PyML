# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal


class Softmax(object):
    """
        Softmax分类器
    """

    def __init__(self):
        """
            构造函数
        """

        w = T.dmatrix('w')  # 权值
        b = T.dvector('b')  # 偏置
        x = T.dmatrix('x')  # 模型的输入
        y = T.dmatrix('y')  # 模型的输出
        n = T.dot(x, w) + b  # 净输出
        p = T.nnet.softmax(n)  # 概率输出

        model_predict = p  # 模型的预测
        object_function = -((y * T.log(p)).sum(axis=1)).mean() + 1e-4 * ((w ** 2).sum() + (b ** 2).sum())
        gradient_vector = T.grad(object_function, [w, b])

        self.f_model_predict = theano.function([x, w, b], model_predict)
        self.f_object_function = theano.function([x, y, w, b], object_function)
        self.f_gradient_vector = theano.function([x, y, w, b], gradient_vector)

        self.datas = None  # 开始训练之前需要绑定训练数据
        self.label = None  # 开始训练之前需要绑定训练标签
        self.parameters = None  # 训练完毕之后需要绑定模型参数，参数顺序为[w,b]

    def do_model_predict(self, x):
        return self.f_model_predict(x, *self.parameters)

    def do_object_function(self, *x):
        return self.f_object_function(self.datas, self.label, *x)

    def do_gradient_vector(self, *x):
        return self.f_gradient_vector(self.datas, self.label, *x)


if __name__ == '__main__':
    # 准备训练数据
    N = 2000
    train_datas = 2.0 * np.random.rand(N, 2) - 1.0
    train_label = np.zeros((N, 4))
    for n in range(N):
        if train_datas[n, 0] >= 0 and train_datas[n, 1] >= 0:
            train_label[n, 0] = 1
        if train_datas[n, 0] < 0 and train_datas[n, 1] >= 0:
            train_label[n, 1] = 1
        if train_datas[n, 0] < 0 and train_datas[n, 1] < 0:
            train_label[n, 2] = 1
        if train_datas[n, 0] >= 0 and train_datas[n, 1] < 0:
            train_label[n, 3] = 1

    model = Softmax()
    model.datas = train_datas
    model.label = train_label

    w = 0.01 * np.random.randn(2, 4)
    b = np.zeros((4,))
    x_optimal, y_optimal = optimal.minimize_GD(model, w, b, max_step=int(1e5), learn_rate=1e-1, epsilon_g=1e-5)
    model.parameters = x_optimal

    predict = model.do_model_predict(train_datas)

    # plt.plot(train_datas,train_label,'r',train_datas,predict,'g')
