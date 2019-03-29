import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
import collections


def step(x,h_,c_):
    return c_+x, h_-x


X = T.fmatrix('X')  # 行代表time_step
output,updates = theano.scan(step,sequences=[X],outputs_info=[T.alloc(np.asarray(0,theano.config.floatX),2),
                                                              T.alloc(np.asarray(0,theano.config.floatX),2)],
                                                n_steps=X.shape[0])

f = theano.function([X],output,updates=updates)

x = np.asarray(np.arange(-3,3).reshape(3,2), theano.config.floatX)
print('x = ', x)

a,b = f(x)



