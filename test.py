# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal

x = T.dmatrix('x')
y = T.dmatrix('y')
z = T.concatenate([x, y], axis=1)

f = theano.function([x, y], z)

a = np.arange(15).reshape(3, 5)
b = np.ones((3, 5))
k = f(a, b)
