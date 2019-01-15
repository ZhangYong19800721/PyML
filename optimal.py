# -*- coding: utf-8 -*-

import sys
import numpy as np
import math

class SINGLE(object):
    """一维线搜索单变量函数包裹器"""
    def __init__(self, F, x0, d0):
        """
            构造函数
            输入：
                F 原始函数
                x0 原始函数的参数（列表，元素为numpy array），一维线搜索的起始点
                d0 搜索方向（列表，元素为numpy array，维度与x0相同）
        """
        self.F = F
        self.x0 = x0
        self.d0 = d0

    def do_object_function(self, m):
        x1 = [x+m*d for m,d in zip(self.x0, self.d0)]
        return self.F.do_object_function(*x1)


class NEGATIVE(object):
    """负向包裹器"""
    def __init__(self, F):
        self.F = F

    def do_object_function(self, *x):
        return -self.F.do_object_function(*x)

    def do_gradient_vector(self, *x):
        return -self.F.do_gradient_vector(*x)

    def do_object_function(self, step_idx, *x):
        return -self.F.do_object_function(step_idx, *x)

    def do_gradient_vector(self, step_idx, *x):
        return -self.F.do_gradient_vector(step_idx, *x)

def minimize_LineSearch_Gold(F, a, b, **options):
    """
        一维精确线性搜索 黄金分割法
        输入：
            F 单变量函数，使用F.do_object_function(x)计算x处的函数值
            a 搜索区间的下界a<b（浮点数）
            b 搜索区间的上界a<b（浮点数）
            options 黄金分割法的参数（字典）
        返回：
            最小点的函数值（浮点数）
            最小点的变量值（浮点数）
    """
    # 设置默认参数
    if 'epsilon' not in options:
        options['epsilon'] = 1e-6 # 当搜索区间足够小时停止算法
        print(f"调用minimize_LineSearch_Gold函数时未指定epsilon参数，将使用默认值{options['epsilon']}")

    # 使用黄金分割法进行一维搜索
    gold = (math.sqrt(5.0)-1.0) / 2.0 # 黄金分割数
    ax = a + (1 - gold)*(b - a) # 计算左侧的黄金分割点
    F_ax = F.do_object_function(ax) # 计算左侧的目标函数值
    bx = a + gold * (b - a) # 计算右侧的黄金分割点
    F_bx = F.do_object_function(bx) # 计算右侧的目标函数值

    while b - a > options['epsilon']: # 当搜索区间宽度下降至epsilon时停止
        if F_ax > F_bx:
            a = ax
            ax = bx
            F_ax = F_bx
            bx = a + gold * (b - a)
            F_bx = F.do_object_function(bx)
        else:
            b = bx
            bx = ax
            F_bx = F_ax
            ax = a + (1 - gold)*(b - a)
            F_ax = F.do_object_function(ax)

    if F_ax > F_bx:
        return F_bx,bx
    else:
        return F_ax,ax

def minimize_LineSearch_Parabola(F,a,b,**options):
    """
        一维精确线性搜索 抛物线法
        参考：马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社）
        输入：
            F 单变量函数，使用F.do_object_function(x)计算x处的函数值
            a 搜索区间的下界a<b（浮点数）
            b 搜索区间的上界a<b（浮点数）
            options 黄金分割法的参数（字典）
        返回：
            最小点的函数值（浮点数）
            最小点的变量值（浮点数）
    """
    # 设置默认参数
    if 'epsilon' not in options:
        options['epsilon'] = 1e-6
        print(f"调用minimize_LineSearch_Parabola函数时未指定epsilon参数，将使用默认值{options['epsilon']}")

    # 找到满足条件的起始点x1,x2,x3,应满足f(x2)<f(x1)且f(x2)<f(x3)
    x1 = a
    x3 = b
    f1 = F.do_object_function(x1)
    f3 = F.do_object_function(x3)

    # 从中间出发向x1逐步靠近，直到找到比f1和f3都小的函数值f2和点x2
    n = 1
    while 1.0 / (2**n) > options['epsilon']:
        n = n + 1
        x2 = x1 + (x3 - x1) / 2**n
        f2 = F.do_object_function(x2)
        if f1 > f2 and f2 < f3:
            break # 当找到满足条件的f2和x2时跳出该循环

    # 如果找不到满足条件的x2
    if f2 > f1 or f2 > f3:
        if f1 < f3:
            return f1,x1
        else:
            return f3,x3

    # 开始搜索
    while min(x3-x1,x2-x1) > options['epsilon']:
        # 计算插值二次函数的最优点
        alfa = (x2**2 - x3**2)*f1 + (x3**2 - x1**2)*f2 + (x1**2 - x2**2)*f3
        beda = (x2**1 - x3**1)*f1 + (x3**1 - x1**1)*f2 + (x1**1 - x2**1)*f3
        if beda == 0:
            break
        else:
            xp = 0.5 * alfa / beda
            if xp <= x1 or xp >= x3:
                break

        # 区间缩小，根据xp与x2，fp与f2的相互关系分六种情况进行区间缩小
        if xp == x2:
            state = 7
            fp = f2
        else:
            fp = F.do_object_function(xp)
            if xp > x2:
                if fp < f2:
                    state = 1
                elif fp > f2:
                    state = 2
                else:
                    state = 3
            elif xp < x2:
                if fp < f2:
                    state = 4
                elif fp > f2:
                    state = 5
                else:
                    state = 6

        if state == 1:
            x1 = x2
            x2 = xp
            f1 = f2
            f2 = fp
        elif state == 2:
            x3 = xp
            f3 = fp
        elif state == 3:
            x1 = x2
            x3 = xp
            x2 = (x1 + x3)/2
            f1 = f2
            f3 = fp
            f2 = F.do_object_function(x2)
        elif state == 4:
            x3 = x2
            x2 = xp
            f3 = f2
            f2 = fp
        elif state == 5:
            x1 = xp
            f1 = fp
        elif state == 6:
            x1 = xp
            x3 = x2
            x2 = (x1 + x3)/2
            f1 = fp
            f3 = f2
            f2 = F.do_object_function(x2)
        elif state == 7:
            x12 = (x2 + x1) / 2
            f12 = F.do_object_function(x12)
            x23 = (x2 + x3) / 2
            f23 = F.do_object_function(x23)
            if f12 <= min(f2,f23):
                x3 = x2
                f3 = f2
                x2 = x12
                f2 = f12
            elif f23 <= min(f2,f12):
                x1 = x2
                f1 = f2
                x2 = x23
                f2 = f23
            elif f2 <= min(f12,f23):
                x1 = x12
                f1 = f12
                x3 = x23
                f3 = f23

    return f2,x2

def minimize_LineSearch_Armijo(F,x,g,d,parameters):
    """
        一维非精确线性搜索 Armijo准则
        参考：马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社）
        输入：
            F 单变量函数，使用F.do_object_function(x)计算x处的函数值
            a 搜索区间的下界a<b（浮点数）
            b 搜索区间的上界a<b（浮点数）
            options 黄金分割法的参数（字典）
        返回：
            最小点的函数值（浮点数）
            最小点的变量值（浮点数）

%       F 函数对象，F.object(x)计算目标函数在x处的值
%       x 搜索的起始位置
%       g 目标函数在x处的梯度
%       d 搜索方向
%       parameters 参数集
%   输出：
%       ny 最优点的函数值
%       nx 最优点的变量值
    """

    %% 参数设置
    if nargin <= 4 % 没有给出参数的情况
        parameters = [];
        % disp('调用armijo函数时没有给出参数集，将使用默认参数');
    end

    if ~isfield(parameters,'beda') % 给出参数但是没有给出beda的情况
        parameters.beda = 0.5;
        % disp(sprintf('调用armijo函数时参数集中没有beda参数，将使用默认值%f',parameters.beda));
    end

    if ~isfield(parameters,'alfa') % 给出参数但是没有给出alfa的情况
        parameters.alfa = 0.2;
        % disp(sprintf('调用armijo函数时参数集中没有alfa参数，将使用默认值%f',parameters.alfa));
    end

    if ~isfield(parameters,'maxs') % 给出参数但是没有给出maxs的情况
        parameters.maxs = 30;
        % disp(sprintf('调用armijo函数时参数集中没有maxs参数，将使用默认值%f',parameters.maxs));
    end

    assert(0 < parameters.beda && parameters.beda <   1);
    assert(0 < parameters.alfa && parameters.alfa < 0.5);
    assert(0 < parameters.maxs);

    %%
    d = 1e3 * d; % 搜索方向的值扩大1000倍，相当于可能的最大学习速度为1000
    m = 0; f = F.object(x);
    while m <= parameters.maxs
        nx = x + parameters.beda^m * d;
        ny = F.object(nx);
        if ny <= f + parameters.alfa * parameters.beda^m * g'* d
            break;
        end
        m = m + 1;
    end
end



def minimize_GD(F, *x0, **options):
    """
        梯度下降算法
        输入：
            F 被最小化的目标函数，使用F.do_object_function(x)计算在x位置的目标函数值，使用F.do_gradient_vector(x)计算在x位置的梯度
            x0 迭代的起始位置，（列表，元素为numpy array）
            options 梯度下降算法参数（字典）
        返回：
            x1 使得目标函数最小化的参数
            y1 当参数为x1时目标函数的值
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
    inc_x = [0.0 * x for x in x0]  # 参数的递增量，初始化为零
    x1 = x0
    y1 = F.do_object_function(*x1)  # 计算目标函数值

    # 开始迭代
    for step in range(1, options['max_step'] + 1):
        g1 = F.do_gradient_vector(*x1)  # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1])  # 计算梯度模
        print("迭代次数：%6d, 目标函数：%10.8f, 梯度模：%10.8f" % (step, y1, ng1))
        if ng1 < options['epsilon_g']:
            break  # 如果梯度足够小就结束迭代
        # 向负梯度方向迭代，并使用动量参数
        inc_x = [options['momentum'] * d - options['learn_rate'] * g for d, g in zip(inc_x, g1)]
        x1 = [x + inc for x, inc in zip(x1, inc_x)]  # 更新参数值
        y1 = F.do_object_function(*x1)  # 计算目标函数值

    return x1, y1


def minimize_SGD(F, *x0, **options):
    """
        随机梯度下降算法
        输入：
            F 被最小化的目标函数，使用F.do_object_function(x)计算在x位置的目标函数值，
              使用F.do_gradient_vector(x)计算在x位置的梯度
            x0 迭代的起始位置，（列表，元素为numpy array）
            options 随机梯度下降算法参数（字典）
        返回：
            x1 使得目标函数最小化的参数
            y1 当参数为x1时目标函数的值
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
        options['epsilon_r'] = 1e-6  # 当学习速率降低至epsilon_r时算法停止
        print(f"调用minimize_SGD函数时未指定epsilon_r参数，将使用默认值{options['epsilon_r']}")

    if 'window' not in options:
        options['window'] = 600  # 计算目标函数均值的滑动窗口大小
        print(f"调用minimize_SGD函数时未指定window参数，将使用默认值{options['window']}")

    # 初始化
    inc_x = [0.0 * x for x in x0]  # 参数的递增量，初始化为零
    learn_rate = options['learn_rate']  # 初始化学习速度
    x1 = x0
    y1 = F.do_object_function(0, *x1)  # 计算目标函数值
    average = y1
    prev_ave = sys.float_info.max

    # 开始迭代
    for step in range(1, options['max_step'] + 1):
        g1 = F.do_gradient_vector(step, *x1)  # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1])  # 计算梯度模
        # 向负梯度方向迭代，并使用动量参数
        inc_x = [options['momentum'] * d - learn_rate * g for d, g in zip(inc_x, g1)]
        x1 = [x + inc for x, inc in zip(x1, inc_x)]  # 更新参数值
        y1 = F.do_object_function(step, *x1)  # 计算目标函数值
        average = (1 - 1 / options['window']) * average + (1 / options['window']) * y1  # 更新滑动均值
        print("迭代次数：%6d, 目标函数：%10.8f, 目标均值：%10.8f, 梯度模：%10.8f, 学习速度：%10.8f" % (step, y1, average, ng1, learn_rate))

        if step % options['window'] == 0:
            if average < prev_ave:
                prev_ave = average
            else:
                learn_rate /= 2.0  # 当目标函数的滑动均值不能降低时，就降低学习速度

        if learn_rate < options['epsilon_r']:
            break  # 如果学习速度足够小就结束算法

    return x1, y1


def minimize_ADAM(F, *x0, **options):
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
        options['epsilon_r'] = 1e-6  # 当学习速率降低至epsilon_r时算法停止
        print(f"调用minimize_ADAM函数时未指定epsilon_r参数，将使用默认值{options['epsilon_r']}")

    if 'window' not in options:
        options['window'] = 600
        print(f"调用minimize_ADAM函数时未指定decay参数，将使用默认值{options['window']}")

    # 初始化
    inc_m = [0.0 * x for x in x0]  # 初始化第1个递增向量
    inc_v = [0.0 * x for x in x0]  # 初始化第2个递增向量
    learn_rate = options['learn_rate']  # 初始化学习速度
    x1 = x0  # 迭代起始点
    y1 = F.do_object_function(0, *x1)  # 计算目标函数值
    average = y1
    prev_ave = sys.float_info.max

    # 开始迭代
    for step in range(1, options['max_step'] + 1):
        g1 = F.do_gradient_vector(step, *x1)  # 计算梯度
        ng1 = sum([np.linalg.norm(g) for g in g1])  # 计算梯度模
        inc_m = [options['beda1'] * m + (1 - options['beda1']) * g for m, g in zip(inc_m, g1)]  # 更新第1个增量向量
        inc_v = [options['beda2'] * v + (1 - options['beda2']) * g ** 2 for v, g in zip(inc_v, g1)]  # 更新第2个增量向量
        inc_mb = [m / (1 - options['beda1'] ** step) for m in inc_m]  # 对第1个增量向量进行修正
        inc_vb = [v / (1 - options['beda2'] ** step) for v in inc_v]  # 对第2个增量向量进行修正
        x1 = [x - learn_rate * m / (np.sqrt(v) + options['epsilon']) for x, m, v in zip(x1, inc_mb, inc_vb)]
        y1 = F.do_object_function(step, *x1)  # 计算目标函数值
        average = (1 - 1 / options['window']) * average + (1 / options['window']) * y1  # 更新滑动均值
        print("迭代次数：%6d, 目标函数：%10.8f, 目标均值：%10.8f, 梯度模：%10.8f, 学习速度：%10.8f" % (step, y1, average, ng1, learn_rate))

        if step % options['window'] == 0:
            if average < prev_ave:
                prev_ave = average
            else:
                learn_rate /= 2.0  # 当目标函数的滑动均值不能降低时，就降低学习速度

        if learn_rate < options['epsilon_r']:
            break  # 如果学习速度足够小就结束算法

    # 返回
    return x1, y1


def minimize_CG(F, *x0, **options):
    """
        共轭梯度法
        输入：
            F 被最小化的目标函数，使用F.do_object_function(x)计算在x位置的目标函数值，使用F.do_gradient_vector(x)计算在x位置的梯度
            x0 迭代的起始位置，（列表，元素为numpy array）
            options 梯度下降算法参数（字典）
        返回：
            x1 使得目标函数最小化的参数
            y1 当参数为x1时目标函数的值
    """

    # 设置默认参数
    if 'max_step' not in options: # 最大迭代次数
        options['max_step'] = int(1e6)
        print(f"调用minimize_CG函数时未指定max_step参数，将使用默认值{options['max_step']}")

    if 'epsilon_g' not in options:
        options['epsilon_g'] = 1e-5 # 最小梯度模（算法的停止条件），当梯度模小于该值时算法停止
        print(f"调用minimize_CG函数时未指定epsilon_g参数，将使用默认值{options['epsilon_g']}")

    if 'reset' not in options:
        options['reset'] = 1000 # 由于存在误差，共轭梯度迭代若干次之后需要重置，当迭代次数到达reset的整数倍时进行重置
        print(f"调用minimize_CG函数时未指定reset参数，将使用默认值{options['reset']}")

    if 'line_search' not in options:
        options['line_search'] = minimize_LineSearch_Gold # 默认的线性搜索器
        print(f"调用minimize_CG函数时未指定line_search参数，将使用默认值{options['line_search']}")

    # 计算起始位置的函数值、梯度、梯度模
    x1 = x0
    y1 = F.do_object_function(*x1)
    g1 = F.do_gradient_vector(*x1)
    ng1 = sum([np.linalg.norm(g) for g in g1])  # 计算梯度模

    # 迭代寻优
    d1 = [-g for g in g1] # 初始搜索方向为负梯度方向
    for step in range(1,options['max_step']+1):
        if ng1 < options['epsilon_g']:
            return x1, y1 # 如果梯度足够小，停止迭代

        # 沿d1方向线搜索
        Fs = SINGLE(F,x1,d1) # 包装为单变量函数
        y2,lamda = options['line_search'](Fs,a,b) # 执行线搜索
        x2 = [x + lamda * d for x,d in zip(x1,d1)] # 迭代到新的位置x2

        if step % options['reset'] == 0 or y1 <= y2: # 当达到重置点或者d1方向不是一个下降方向
            d1 = -g1 # 重新设定搜索方向为负梯度方向
            Fs = SINGLE(F,x1,d1) # 包装为单变量函数
            y2,lamda = options['line_search'](Fs,a,b) # 执行线搜索
            x2 = [x + lamda * d for x,d in zip(x1,d1)] # 迭代到新的位置x2
            g2 = F.do_gradient_vector(*x2) # 计算x2位置的梯度
            d2 = [-g for g in g2] # 搜索方向
            ng2 = sum([np.linalg.norm(g) for g in g2]) # 计算梯度模ng2
            x1,d1,g1,y1,ng1 = x2,d2,g2,y2,ng2
            print("迭代次数：%6d, 目标函数：%10.8f, 梯度模：%10.8f" % (step, y1, ng1))
            continue

        g2 = F.do_gradient_vector(*x2) # 计算x2位置的梯度
        ng2 = sum([np.linalg.norm(g) for g in g2]) # 计算梯度模ng2
        beda = sum([q2.reshape(1,-1).dot((q2-q1).reshape(-1,1)) for q1,q2 in zip(g1,g2)]) / sum([g.reshape(1,-1).dot(g.reshape(-1,1)) for g in g1]) # g2'*(g2-g1)/(g1'*g1)
        d2 = [-g + beda * d for g,d in zip(g2,d1)] # 计算x2处的搜索方向d2
        x1,d1,g1,y1,ng1 = x2,d2,g2,y2,ng2
        print("迭代次数：%6d, 目标函数：%10.8f, 梯度模：%10.8f" % (step, y1, ng1))