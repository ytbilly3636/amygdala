# -*- coding: utf-8 -*-

from amygdala.layer import LateralNucleus as LA
from amygdala.layer import CentralNucleus as CE

import numpy as np
import cv2
import six

class Amygdala(object):
    def __init__(self):
        self.la1 = LA(la_size=1, la_map_size=(8, 8), la_in_size=3*64*64)
        self.la2 = LA(la_size=1, la_map_size=(8, 8), la_in_size=3)
        self.ce = CE(in_size=2*8*8, out_size=3)
        
    def inference(self, x1, x2, var=0.4):
        h1 = self.la1.inference(x1, var)
        h2 = self.la2.inference(x2, var)
        h = np.concatenate((h1, h2), axis=1)
        y = self.ce.inference(h)
        return y
        
    def update(self, t, lr_la=0.01, var_la=0.5, lr_ce=0.1):
        self.la1.update(lr_la, var_la)
        self.la2.update(lr_la, var_la)
        self.ce.update(t, lr_ce)

amy = Amygdala()

def pretraining():
    dammy_t = np.zeros((1, 3))

    for i in six.moves.range(200):
        x1 = [np.random.rand(1, 3*64*64)]
        x2 = [np.eye(3)[int(np.random.randint(3))].reshape(1, 3)]
        amy.inference(x1, x2)
        amy.update(dammy_t, lr_la=0.1, var_la=1.5, lr_ce=0.0)
        
        img = cv2.resize(amy.la2.soms[0].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('som', img)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)

x1s = np.random.rand(3, 1, 3*64*64)
        
def could_you_get_that(x1, x2, t, lr):
    x1 = [x1s[x1]]
    x2 = [np.eye(3)[x2].reshape(1, 3)]
    t = np.eye(3)[t].reshape(1, -1)
    y = amy.inference(x1, x2)
    amy.update(t, lr_ce=lr)
    return y[0]
    
    
print('pretraining')
pretraining()

# Person 0 at place 0 order 0
print('A:cookie')
for i in six.moves.range(5):
    print(could_you_get_that(0, 0, 0, 1.0))

# Person 0 at place 1 order 1
print('A:potato chips')
for i in six.moves.range(5):
    print(could_you_get_that(0, 1, 1, 1.0))
    
# Person 0 at place 2 order 2  
print('A:potato stick')
for i in six.moves.range(5):
    print(could_you_get_that(0, 2, 2, 1.0))
    
# Person 1 at place 2 order 0
print('B:cookie')
for i in six.moves.range(5):
    print(could_you_get_that(0, 0, 0, 1.0))

# Person 1 at place 0 order 1
print('B:potato chips')
for i in six.moves.range(5):
    print(could_you_get_that(0, 1, 1, 1.0))
    
# Person 1 at place 1 order 2  
print('B:potato stick')
for i in six.moves.range(5):
    print(could_you_get_that(0, 2, 2, 1.0))
