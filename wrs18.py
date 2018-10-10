# -*- coding: utf-8 -*-

from amygdala.layer import LateralNucleus as LA
from amygdala.layer import CentralNucleus as CE

import numpy as np
import cv2
import six

class Amygdala(object):
    def __init__(self):
        self.la1 = LA(la_size=1, la_map_size=(8, 8), la_in_size=3*64*64)    # for face
        self.la2 = LA(la_size=2, la_map_size=(8, 8), la_in_size=3)          # for place & time
        self.ce = CE(in_size=3*8*8, out_size=3)
        
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

face = np.load('data/face.npz')
face_buri = face['buri'].reshape(1, 100, -1).astype(np.float32) / 255.0
face_miyo = face['miyo'].reshape(1, 100, -1).astype(np.float32) / 255.0
face_sika = face['sika'].reshape(1, 100, -1).astype(np.float32) / 255.0
fs = np.append(face_buri, face_miyo, axis=0)
fs = np.append(fs, face_sika, axis=0)

def pretraining():
    dammy_t = np.zeros((1, 3))

    for i in six.moves.range(5000):
        x1 = [fs[int(np.random.randint(3))][int(np.random.randint(100))].reshape(1, -1)]    # face
        x2 = [np.eye(3)[int(np.random.randint(3))].reshape(1, -1)]                          # place
        x2.append(np.eye(3)[int(np.random.randint(3))].reshape(1, -1))                       # time
        amy.inference(x1, x2)
        amy.update(dammy_t, lr_la=0.1, var_la=1.2, lr_ce=0.0)
        
        img_f = np.zeros((64*8, 64*8, 3))
        for row in six.moves.range(8):
            for col in six.moves.range(8):
                img_f[row*64:(row+1)*64, col*64:(col+1)*64] = amy.la1.soms[0].w[row][col].reshape(64, 64, 3)
        
        img_p = cv2.resize(amy.la2.soms[0].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        img_t = cv2.resize(amy.la2.soms[1].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('face', img_f)
        cv2.imshow('place', img_p)
        cv2.imshow('time', img_t)
        cv2.waitKey(1)
    
    cv2.imwrite('face.png', img_f * 255)
    cv2.imwrite('place.png', img_p * 255)
    cv2.imwrite('time.png', img_t * 255)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
        
def could_you_get_that(x1, x2, x3, t, lr):
    x1 = [fs[x1][0].reshape(1, -1)]
    x2 = [np.eye(3)[x2].reshape(1, -1)]
    x2.append(np.eye(3)[x3].reshape(1, -1))
    t = np.eye(3)[t].reshape(1, -1)
    y = amy.inference(x1, x2)
    print('\tface:', amy.la1.soms[0].c_buf)
    print('\tplace:', amy.la2.soms[0].c_buf)
    print('\ttime:', amy.la2.soms[1].c_buf)
    amy.update(t, lr_ce=lr)
    return y[0]
    
    
print('pretraining')
pretraining()

print('Person 0 in place 0 order 0 at 0')
for i in six.moves.range(5):
    print(could_you_get_that(0, 0, 0, 0, 1.5))

print('Person 0 in place 1 order 1 at 1')
for i in six.moves.range(5):
    print(could_you_get_that(0, 1, 1, 1, 1.5))
    
print('Person 0 in place 0 order 0 at 0')
print(could_you_get_that(0, 0, 0, 0, 0.0))

print('Person 0 in place 1 order 1 at 1')
print(could_you_get_that(0, 1, 1, 1, 0.0))
