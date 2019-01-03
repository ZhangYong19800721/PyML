# -*- coding: utf-8 -*-

import numpy as np

def minimize_GD(F,*x0,**options):
    """
        梯度下降算法
            F 被最小化的目标函数，使用F.do_object_function(x)计算在x位置的目标函数值，使用F.do_gradient_vector(x)计算在x位置的梯度
            x0 迭代的起始位置，（列表，元素为numpy array）
            options 梯度下降算法参数（字典）
    """
    # 设置默认参数
    if 'learn_rate' not in options:
        options['learn_rate'] = 1e-5
        print(f"调用minimize_GD函数时未指定learn_rate参数，将使用默认值{options['learn_rate']}")
    
    if 'momentum' not in options:
        options['momentum'] = 0.9
        print(f"调用minimize_GD函数时未指定momentum参数，将使用默认值{options['momentum']}")
    
    if 'max_step' not in options:
        options['max_step'] = 10000
        print(f"调用minimize_GD函数时未指定max_step参数，将使用默认值{options['max_step']}")
        
    if 'epsilon_g' not in options:
        options['epsilon_g'] = 1e-5
        print(f"调用minimize_GD函数时未指定epsilon_g参数，将使用默认值{options['epsilon_g']}")
 
    # 初始化
    inc_x = [0.0 * x for x in x0] # 参数的递增量，初始化为零
    x1 = x0  
    y1 = F.do_object_function(*x1) # 计算目标函数值
    
    # 开始迭代
    for step in range(1,options['max_step']+1):
        g1 = F.do_gradient_vector(*x1) # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1]) # 计算梯度模
        print("迭代次数：%6d, 目标函数：%10.6f, 梯度模：%10.6f" % (step,y1,ng1))
        if ng1 < options['epsilon_g']:
            break # 如果梯度足够小就结束迭代
        # 向负梯度方向迭代，并使用动量参数
        inc_x = [options['momentum'] * d - (1 - options['momentum']) * options['learn_rate'] * g for d,g in zip(inc_x,g1)] 
        x1 = [x + inc for x,inc in zip(x1,inc_x)] # 更新参数值
        y1 = F.do_object_function(*x1) # 计算目标函数值
    
    return x1,y1

