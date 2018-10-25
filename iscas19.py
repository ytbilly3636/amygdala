# -*- coding: utf-8 -*-

from amygdala.layer import LateralNucleus as LA
from amygdala.layer import CentralNucleus as CE

import numpy as np
import cv2
import six
import os

# data
faces = []
files = os.listdir('data/img')
files = sorted(files)
for f in files:
    img = cv2.imread('data/img/' + f)
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    faces.append(img)

# amygdala model
class Amygdala(object):
    def __init__(self):
        self.la1 = LA(la_size=1, la_map_size=(8, 8), la_in_size=3*64*64)    # face
        self.la2 = LA(la_size=1, la_map_size=(8, 8), la_in_size=2)          # place
        self.ce = CE(in_size=2*8*8, out_size=4)
        
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

# func: pretraining
def pretraining():
    dammy_t = np.zeros((1, 4))

    for i in six.moves.range(1000):
        x1 = [faces[int(np.random.randint(7))].reshape(1, 3*64*64)]
        x2 = [np.eye(2)[int(np.random.randint(2))].reshape(1, 2)]
        amy.inference(x1, x2)
        amy.update(dammy_t, lr_la=0.1, var_la=1.5, lr_ce=0.0)
        
        img_f = np.zeros((64*8, 64*8, 3))
        for row in six.moves.range(8):
            for col in six.moves.range(8):
                img_f[row*64:(row+1)*64, col*64:(col+1)*64] = amy.la1.soms[0].w[row][col].reshape(64, 64, 3)
        cv2.imshow('face', img_f)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)

# func: training      
def could_you_get_that(x1, x2, t, lr):
    x1 = [faces[x1].reshape(1, 3*64*64)]
    x2 = [np.eye(2)[x2].reshape(1, 2)]
    t = np.eye(4)[t].reshape(1, 4)
    y = amy.inference(x1, x2)
    amy.update(t, lr_ce=lr)
    return y[0]
    
    
print('pretraining')
pretraining()

# Person 0 at place 0 order 0
print('A')
for i in six.moves.range(10):
    print(could_you_get_that(0, 0, 0, 1.2))

# Person 0 at place 1 order 1
print('B')
for i in six.moves.range(10):
    print(could_you_get_that(0, 1, 1, 1.2))
    
# Person 4 at place 0 order 2  
print('C')
for i in six.moves.range(10):
    print(could_you_get_that(4, 0, 2, 1.2))
    
# Person 4 at place 1 order 3
print('D')
for i in six.moves.range(10):
    print(could_you_get_that(4, 1, 3, 1.2))

print('A')
print(could_you_get_that(0, 0, 0, 0.0))
print('B')
print(could_you_get_that(0, 1, 1, 0.0))
print('C')
print(could_you_get_that(4, 0, 2, 0.0))
print('D')
print(could_you_get_that(4, 1, 3, 0.0))
