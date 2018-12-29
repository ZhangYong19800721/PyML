# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
import minimize

x,y = T.dscalars('x','y')
a = theano.shared(1.0,name='a')

model = a*x
f_model = theano.function([x],model)
print(f_model(4))

cost = 0.5 * (model - y)**2
f_cost = theano.function([x,y],cost)
print(f_cost(4,5))

minimize.GradientDescend(cost,x,y,[a],4,16)
