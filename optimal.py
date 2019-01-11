# -*- coding: utf-8 -*-

import sys
import numpy as np
import math

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
        print("迭代次数：%6d, 目标函数：%10.8f, 梯度模：%10.8f" % (step,y1,ng1))
        if ng1 < options['epsilon_g']:
            break # 如果梯度足够小就结束迭代
        # 向负梯度方向迭代，并使用动量参数
        inc_x = [options['momentum'] * d - options['learn_rate'] * g for d,g in zip(inc_x,g1)] 
        x1 = [x + inc for x,inc in zip(x1,inc_x)] # 更新参数值
        y1 = F.do_object_function(*x1) # 计算目标函数值
    
    return x1,y1


def minimize_SGD(F,*x0,**options):
    """
        随机梯度下降算法
            F 被最小化的目标函数，使用F.do_object_function(x)计算在x位置的目标函数值，
              使用F.do_gradient_vector(x)计算在x位置的梯度
            x0 迭代的起始位置，（列表，元素为numpy array）
            options 梯度下降算法参数（字典）
    """
    # 设置默认参数
    if 'learn_rate' not in options:
        options['learn_rate'] = 1e-5
        print(f"调用minimize_SGD函数时未指定learn_rate参数，将使用默认值{options['learn_rate']}")
    
    if 'momentum' not in options:
        options['momentum'] = 0.9
        print(f"调用minimize_SGD函数时未指定momentum参数，将使用默认值{options['momentum']}")
    
    if 'max_step' not in options:
        options['max_step'] = 10000
        print(f"调用minimize_SGD函数时未指定max_step参数，将使用默认值{options['max_step']}")
        
    if 'epsilon_r' not in options:
        options['epsilon_r'] = 1e-6 # 当学习速率降低至epsilon_r时算法停止
        print(f"调用minimize_SGD函数时未指定epsilon_r参数，将使用默认值{options['epsilon_r']}")
        
    if 'window' not in options:
        options['window'] = 600 # 计算目标函数均值的滑动窗口大小
        print(f"调用minimize_SGD函数时未指定window参数，将使用默认值{options['window']}")
 
    # 初始化
    inc_x = [0.0 * x for x in x0] # 参数的递增量，初始化为零
    learn_rate = options['learn_rate'] # 初始化学习速度
    x1 = x0  
    y1 = F.do_object_function(0,*x1) # 计算目标函数值
    average = y1 
    prev_ave = sys.float_info.max
    
    # 开始迭代
    for step in range(1,options['max_step']+1):
        g1 = F.do_gradient_vector(step,*x1) # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1]) # 计算梯度模
        # 向负梯度方向迭代，并使用动量参数
        inc_x = [options['momentum'] * d - learn_rate * g for d,g in zip(inc_x,g1)] 
        x1 = [x + inc for x,inc in zip(x1,inc_x)] # 更新参数值
        y1 = F.do_object_function(step,*x1) # 计算目标函数值
        average = (1-1/options['window']) * average + (1/options['window'])*y1 # 更新滑动均值
        print("迭代次数：%6d, 目标函数：%10.8f, 目标均值：%10.8f, 梯度模：%10.8f, 学习速度：%10.8f" % (step,y1,average,ng1,learn_rate))
        
        if step % options['window'] == 0:
            if average < prev_ave:
                prev_ave = average
            else:
                learn_rate /= 2.0 # 当目标函数的滑动均值不能降低时，就降低学习速度
        
        if learn_rate < options['epsilon_r']:
            break # 如果学习速度足够小就结束算法
            
    return x1,y1

def minimize_ADAM(F,*x0,**options):
    """
        ADAM 随机梯度下降
        参考文献“ADAM:A Method For Stochastic Optimization”,2014
        输入：
            F 模型。调用 F.do_gradient_vector(step,*x1) 计算目标函数的梯度，
                   调用 F.do_object_function(step,*x1) 计算目标函数的值，
                   其中 step 指示迭代步数，即指示使用哪个 minibatch
            x0 迭代的起始位置
            options 最优化参数集
        输出：
            x 最优的参数解
            y 最小的函数值
    """
    
    # 设置默认参数
    if 'epsilon' not in options:
        options['epsilon'] = 1e-8 
        print(f"调用minimize_ADAM函数时未指定epsilon参数，将使用默认值{options['epsilon']}")
        
    if 'omiga' not in options:
        options['omiga'] = 1e-3
        print(f"调用minimize_ADAM函数时未指定omiga参数，将使用默认值{options['omiga']}")
        
    if 'max_step' not in options:
        options['max_step'] = int(1e6)
        print(f"调用minimize_ADAM函数时未指定max_step参数，将使用默认值{options['max_step']}")
    
    if 'learn_rate' not in options:
        options['learn_rate'] = 1e-2
        print(f"调用minimize_ADAM函数时未指定learn_rate参数，将使用默认值{options['learn_rate']}")
    
    if 'beda1' not in options:
        options['beda1'] = 0.9
        print(f"调用minimize_ADAM函数时未指定beda1参数，将使用默认值{options['beda1']}")
        
    if 'beda2' not in options:
        options['beda2'] = 0.999
        print(f"调用minimize_ADAM函数时未指定beda1参数，将使用默认值{options['beda2']}")
        
    if 'epsilon_r' not in options:
        options['epsilon_r'] = 1e-6 # 当学习速率降低至epsilon_r时算法停止
        print(f"调用minimize_ADAM函数时未指定epsilon_r参数，将使用默认值{options['epsilon_r']}")
        
    if 'window' not in options:
        options['window'] = 0
        print(f"调用minimize_ADAM函数时未指定decay参数，将使用默认值{options['window']}")
        
    # 初始化
    inc_m = [0.0 * x for x in x0] # 初始化第1个递增向量
    inc_v = [0.0 * x for x in x0] # 初始化第2个递增向量
    learn_rate = options['learn_rate'] # 初始化学习速度
    x1 = x0  # 迭代起始点
    y1 = F.do_object_function(0,x1) # 计算目标函数值
    average = y1 
    prev_ave = sys.float_info.max
    
    # 开始迭代
    for step in range(1,options['max_step']+1):
        g1 = F.do_gradient_vector(step,x1) # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1]) # 计算梯度模
        inc_m = [options['beda1'] * m + (1 - options['beda1']) * g    for m,g in zip(inc_m,g1)] # 更新第1个增量向量
        inc_v = [options['beda2'] * v + (1 - options['beda2']) * g**2 for v,g in zip(inc_v,g1)] # 更新第2个增量向量
        inc_mb = [m / (1 - options['beda1']**step) for m in inc_m] # 对第1个增量向量进行修正
        inc_vb = [v / (1 - options['beda2']**step) for v in inc_v] # 对第2个增量向量进行修正      
        x1 = [x - learn_rate * m / (np.sqrt(v) + options['epsilon']) for x,m,v in zip(x1,inc_mb,inc_vb)]
        y1 = F.do_object_function(step,x1) # 计算目标函数值
        average = (1-1/options['window']) * average + (1/options['window'])*y1 # 更新滑动均值
        print("迭代次数：%6d, 目标函数：%10.8f, 目标均值：%10.8f, 梯度模：%10.8f, 学习速度：%10.8f" % (step,y1,average,ng1,learn_rate))
        
        if step % options['window'] == 0:
            if average < prev_ave:
                prev_ave = average
            else:
                learn_rate /= 2.0 # 当目标函数的滑动均值不能降低时，就降低学习速度
        
        if learn_rate < options['epsilon_r']:
            break # 如果学习速度足够小就结束算法
    
    # 返回
    return x1,y1
