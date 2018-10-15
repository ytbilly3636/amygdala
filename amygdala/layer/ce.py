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
    import time

    ce = CentralNucleus(in_size=8, out_size=3)
    
    x0 = np.random.rand(1, 8)
    t0 = np.eye(3)[0].reshape(1, -1)
    
    s1 = time.time()
    y = ce.inference(x0)
    ce.update(t0, 0.01)
    print time.time() - s1
