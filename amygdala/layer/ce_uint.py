# -*- coding: utf-8 -*-

from ce import CentralNucleus
import numpy as np

class CentralNucleusUint(CentralNucleus):
    def __init__(self, in_size, out_size):
        super(CentralNucleusUint, self).__init__(in_size, out_size)
        self.w = (self.w * 255).astype(np.uint8)
        
    def inference(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.w.shape[1]:
            import sys; sys.exit('x.shape must be (batch_size, in_size)')
        
        self.x_buf = copy.deepcopy(x)
        self.y_buf = x.dot(self.w.T)
        return self.y_buf
        
    def update(self, t, sft):
        if len(t.shape) != 2 or t.shape[1] != self.w.shape[0]:
            import sys; sys.exit('t.shape must be (batch_size, out_size)')
    
        y_max = 255.0 ** 2 * self.w.shape[1]
        if np.max(t) != y_max and np.max(t) == 0:
            t *= y_max / np.max(t)
            t = t.astype(np.int32)
    
        w = self.w.astype(np.int32)
        delta = self.__bitshift((self.y_buf - t).T.dot(self.x_buf), sft) / t.shape[0]
        w -= delta
        self.w -= np.clip(w, 0, 255).astype(np.uint8)
        
    def __bitshift(self, x, sft):
        pos = np.where(x >= 0)
        neg = np.where(x < 0)
        x[pos] = x[pos] >> sft
        x[neg] = -x[pos] >> sft
        x[neg] = -x[neg]
        return x
        

