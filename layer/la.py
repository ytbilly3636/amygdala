# -*- coding: utf-8 -*-

import numpy as np
import six
import copy

class SOM(object):
    def __init__(self, map_size, in_size):
        if len(map_size) != 2:
            import sys; sys.exit('map_size must be 2-dim list, tuple or array')        
        self.w = np.random.rand(map_size[0], map_size[1], in_size)
        
    def inference(self, x, var=0.4):
        self.x_buf = x.reshape(-1, self.w.shape[2])
        sim = self.__cosine_similarity(self.x_buf)
        self.c_buf = self.__winner_indices(sim)
        return self.__gaussian(self.c_buf, var)

    def update(self, lr, var=1.0):
        h = self.__gaussian(self.c_buf, var)
        dw = np.tensordot(h, self.x_buf, axes=(0, 0)) - self.w * np.sum(h, axis=0).reshape(self.w.shape[0], self.w.shape[1], 1)
        self.w += lr * dw / self.x_buf.shape[0]

    def __cosine_similarity(self, x):
        x_size = np.sum(x ** 2, axis=1).reshape(-1, 1)
        w_size = np.sum(self.w ** 2, axis=2).reshape(self.w.shape[0], self.w.shape[1], 1)
        cos = np.tensordot(x / x_size, self.w / w_size, axes=(1, 2))
        return cos
        
    def __winner_indices(self, x):
        x = x.reshape(x.shape[0], -1)
        c = np.argmax(x, axis=1)
        cy = c / self.w.shape[0]
        cx = c % self.w.shape[0]
        return np.append(cy.reshape(-1, 1), cx.reshape(-1, 1), axis=1)
        
    def __gaussian(self, c, var):
        indices = np.arange(c.shape[0]*self.w.shape[0]*self.w.shape[1]).reshape(c.shape[0], self.w.shape[0], self.w.shape[1])
        y = indices / self.w.shape[1] % self.w.shape[0]
        x = indices % self.w.shape[1]
        dy = np.abs(y - c[:,0].reshape(-1, 1, 1))
        dx = np.abs(x - c[:,1].reshape(-1, 1, 1))
        return np.exp(-(dy**2 + dx**2) / (var**2*2))


class LateralNucleus(object):
    def __init__(self, la_size, la_map_size, la_in_size):
        self.soms = [SOM(map_size=la_map_size, in_size=la_in_size) for i in six.moves.range(la_size)]
        
    def inference(self, xs, var):
        if len(xs) != len(self.soms):
            import sys; sys.exit('xs must be <la_size>-dim lists')        
        batch_size = xs[0].shape[0]            
        y = self.soms[0].inference(xs[0], var).reshape(batch_size, -1)
        for l in six.moves.range(1, len(self.soms)):
            y = np.concatenate((y, self.soms[l].inference(xs[l], var).reshape(batch_size, -1)), axis=1)        
        return y
        
    def update(self, lr, var):
        for l in six.moves.range(len(self.soms)):
            self.soms[l].update(lr, var)

        
if __name__ == '__main__':
    import cv2
    
    la = LateralNucleus(la_size=4, la_map_size=(10, 10), la_in_size=3)
    for i in six.moves.range(10000):
        xs = [np.random.rand(3).reshape(1, 3) for i in six.moves.range(4)]
        la.inference(xs, var=0.4)
        la.update(lr=0.1, var=1.5)
        
        pic0 = cv2.resize(la.soms[0].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        pic1 = cv2.resize(la.soms[1].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        pic2 = cv2.resize(la.soms[2].w, (200, 200), interpolation=cv2.INTER_NEAREST)
        pic3 = cv2.resize(la.soms[3].w, (200, 200), interpolation=cv2.INTER_NEAREST)        
        cv2.imshow('0', pic0)
        cv2.imshow('1', pic1)
        cv2.imshow('2', pic2)
        cv2.imshow('3', pic3)
        cv2.waitKey(1)
