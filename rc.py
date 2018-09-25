# -*- coding: utf-8 -*-

from amygdala.layer import LateralNucleusRecurrent as LARC
from amygdala.layer import CentralNucleus as CE

import numpy as np
import cv2
import six

class AmygdalaRC(object):
    def __init__(self):
        self.la = LARC(la_size=1, la_map_size=(8, 8), la_in_size=3)
        self.ce = CE(in_size=1*8*8, out_size=2)
    
    def reset(self):
        self.la.reset()
        
    def inference(self, xs, var=0.5, beta=0.75):
        h = self.la.inference(xs, var, beta)
        y = self.ce.inference(h)
        return y
        
    def update(self, t, lr_la=0.0, var_la=0.1, lr_ce=1.0):
        self.la.update(lr_la, var_la)
        self.ce.update(t, lr_ce)

amy = AmygdalaRC()
amy.reset()

def pretraining():
    dammy_t = np.zeros((1, 2))

    for i in six.moves.range(1500):
        x = [np.random.rand(1, 3)]
        amy.inference(x)
        amy.update(dammy_t, lr_la=0.1, var_la=1.0, lr_ce=0.0)
        
        img = cv2.resize(amy.la.soms[0].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('som', img)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
print('pretraining')
pretraining()


r = [np.array([[0.0, 0.0, 1.0]])]
g = [np.array([[0.0, 1.0, 0.0]])]
b = [np.array([[1.0, 0.0, 0.0]])]
t = np.eye(2)[0].reshape(1, -1)

for i in six.moves.range(10):
    print('---', i, '---')
    amy.reset()
    print('reset')
    print('r', amy.inference(r)[0])
    print('g', amy.inference(g)[0])
    print('b', amy.inference(b)[0])
    amy.update(t)
    
    amy.reset()
    print('reset')
    print('r', amy.inference(r)[0])
    print('b', amy.inference(b)[0])
    print('g', amy.inference(g)[0])
    
    amy.reset()
    print('reset')
    print('g', amy.inference(g)[0])
    print('r', amy.inference(r)[0])
    print('b', amy.inference(b)[0])
    
    amy.reset()
    print('reset')
    print('g', amy.inference(g)[0])
    print('b', amy.inference(b)[0])
    print('r', amy.inference(r)[0])
    
    amy.reset()
    print('reset')
    print('b', amy.inference(b)[0])
    print('r', amy.inference(r)[0])
    print('g', amy.inference(g)[0])
    
    amy.reset()
    print('reset')
    print('b', amy.inference(b)[0])
    print('g', amy.inference(g)[0])
    print('r', amy.inference(r)[0])
    
    print('------')
