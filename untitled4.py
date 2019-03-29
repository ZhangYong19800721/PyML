import theano
import theano.tensor as T
import numpy as np

vector1 = T.vector('vector1')
vector2 = T.vector('vector2')

output, updates = theano.scan(fn=lambda a, b : a * b, sequences=[vector1, vector2])


f = theano.function(inputs=[vector1, vector2], outputs=output, updates=updates)

vector1_value = np.arange(0, 5).astype(theano.config.floatX) # [0,1,2,3,4]
vector2_value = np.arange(1, 6).astype(theano.config.floatX) # [1,2,3,4,5]
print(f(vector1_value, vector2_value))