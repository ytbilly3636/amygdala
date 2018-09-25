# -*- coding: utf-8 -*-

import numpy as np
from ce import CentralNucleus

class CentralNucleusLVQ(CentralNucleus):
    def update(self, t, lr):
        if len(t.shape) != 2 or t.shape[1] != self.w.shape[0]:
            import sys; sys.exit('t.shape must be (batch_size, out_size)')
    
        delta = lr * (t.T.dot(self.x_buf) - self.w) / t.shape[0]
        self.w += delta
