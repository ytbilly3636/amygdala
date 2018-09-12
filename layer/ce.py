# -*- coding: utf-8 -*-

import numpy as np
import copy

class CentralNucleus(object):
    def __init__(self, in_size, out_size):
        self.w = np.random.rand(out_size, in_size)
        
    def inference(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.w.shape[1]:
            import sys; sys.exit('x.shape must be (batch_size, in_size)')
        
        self.x_buf = copy.deepcopy(x)
        self.y_buf = self.__softmax(x.dot(self.w.T))
        return self.y_buf
        
    def update(self, t, lr):
        if len(t.shape) != 2 or t.shape[1] != self.w.shape[0]:
            import sys; sys.exit('t.shape must be (batch_size, out_size)')
    
        delta = lr * ((self.y_buf - t) * self.__d_softmax(self.y_buf)).T.dot(self.x_buf) / t.shape[0]
        self.w -= delta
    
    def __softmax(self, x):
        y = np.exp(x)
        y = y / np.sum(y, axis=1)
        return y
        
    def __d_softmax(self, x):
        return x * (1 - x)  
              
        
if __name__ == '__main__':
    import six

    ce = CentralNucleus(in_size=100, out_size=2)
    
    x0 = np.random.rand(1, 100)
    x1 = np.random.rand(1, 100)
    t0 = np.eye(2)[0].reshape(1, -1)
    t1 = np.eye(2)[1].reshape(1, -1)
    
    for i in six.moves.range(10):
        print('x0:', ce.inference(x0))
        ce.update(t0, lr=0.1)
        print('x1:', ce.inference(x1))
        ce.update(t1, lr=0.1)
