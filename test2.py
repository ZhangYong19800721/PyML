# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

A = T.dmatrix('A')
B = T.dmatrix('B')
y = T.dot(A,B)
f = theano.function([A,B],y)

print(f([[2,1]],[[1,2],[3,4]]))