import numpy as np
import theano
import theano.tensor as T

x = T.dscalar('x')
y = 2*x
f = theano.function([x],y)
print(f(1))