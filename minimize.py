# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

def GradientDescend(F,D,L,W,datas,label,**options):
    """
        梯度下降算法
            F 被最小化的目标函数
            D 数据参数（符号对象）
            L 标签参数（符号对象）
            W 模型参数（符号对象列表）
            datas 训练数据（numpy array）
            label 训练标签（numpy array）
            w0 迭代的起始点（列表）
            options 梯度下降算法参数（字典）
    """
    # 设置默认参数
    if 'learn_rate' not in options:
        options['learn_rate'] = 1e-4
        print(f"调用GradientDescend函数是未指定learn_rate参数，将使用默认值{options['learn_rate']}")
    
    if 'momentum' not in options:
        options['momentum'] = 0.9
        print(f"调用GradientDescend函数是未指定momentum参数，将使用默认值{options['momentum']}")
    
    if 'max_step' not in options:
        options['max_step'] = 10000
        print(f"调用GradientDescend函数是未指定max_step参数，将使用默认值{options['max_step']}")
        
    if 'epsilon_g' not in options:
        options['epsilon_g'] = 1e-3
        print(f"调用GradientDescend函数是未指定epsilon_g参数，将使用默认值{options['epsilon_g']}")
 
    G = T.grad(F,W) # 计算目标函数对W参数的梯度
    #INC = [-options['learn_rate']*g for g in G]
    U = [(w,w - options['learn_rate']*g) for w,g in zip(W,G)] # 权值更新规则
    GN = sum([(g**2).sum() for g in G]) # 梯度模
    train = theano.function([D,L],[F,GN],updates=U)
    for step in range(options['max_step']):
        object_value, gradient_norm = train(datas,label)
        print(f"迭代次数：{step}, 目标函数：{object_value}, 梯度模：{gradient_norm}")
        if gradient_norm < options['epsilon_g']:
            break