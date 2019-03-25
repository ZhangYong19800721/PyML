# -*- coding: utf-8 -*-
"""
    多层感知器（MLP）
    基于theano实现，使用GPU训练
"""

import numpy as np
import theano
import theano.tensor as T
import collections
from optimizer import minimize


class MLP(object):
    def __init__(self, *dim):
        
        self.parameters = collections.OrderedDict()  # 模型的参数集合


if __name__ == "__main__":
    print("Test")
