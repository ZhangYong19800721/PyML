# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import optimal

x = T.dvector('x')
y = T.dmatrix('y')
z = x.dot(y)

f = theano.function([x, y], z)

a = np.arange(4).reshape((4,))
b = np.ones((4, 4))
print(a)
print(b)
k = f(a, b)
print(k)
print(k+b)
