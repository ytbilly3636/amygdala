# -*- coding: utf-8 -*-

from layer import LateralNucleus as LA
from layer import CentralNucleus as CE

import numpy as np
import cv2
import six

class Amygdala(object):
    def __init__(self):
        self.la = LA(la_size=1, la_map_size=(8, 8), la_in_size=3)
        self.ce = CE(in_size=1*8*8, out_size=2)
        
    def inference(self, xs, var=0.4):
        h = self.la.inference(xs, var)
        y = self.ce.inference(h)
        return y
        
    def update(self, t, lr_la=0.01, var_la=0.5, lr_ce=0.1):
        self.la.update(lr_la, var_la)
        self.ce.update(t, lr_ce)

amy = Amygdala()

def pretraining():
    dammy_t = np.zeros((1, 2))

    for i in six.moves.range(1000):
        x = [np.random.rand(1, 3)]
        amy.inference(x)
        amy.update(dammy_t, lr_la=0.1, var_la=1.0, lr_ce=0.0)
        
        img = cv2.resize(amy.la.soms[0].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('som', img)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
        
def learn_green():
    x = [np.array([[0.0, 1.0, 0.0]])]
    t = np.eye(2)[0].reshape(1, -1)
    y = amy.inference(x)
    amy.update(t)
    return np.argmax(y[0])
    
def learn_red():
    x = [np.array([[0.0, 0.0, 1.0]])]
    t = np.eye(2)[1].reshape(1, -1)
    y = amy.inference(x)
    amy.update(t)
    return np.argmax(y[0])
    
print('pretraining')
pretraining()
for i in six.moves.range(20):
    print('green', learn_green())
    print('red', learn_red())
    
