# -*- coding: utf-8 -*-

import numpy as np
from .la import LateralNucleus

class LateralNucleusRecurrent(LateralNucleus):
    def reset(self):
        self.y_buf = np.zeros(1)

    def inference(self, xs, var, beta):
        y = super(LateralNucleusRecurrent, self).inference(xs, var)

        if self.y_buf.all() == 0:
            self.y_buf = y
        else:
            self.y_buf = beta * self.y_buf + y

        return self.y_buf


if __name__ == '__main__':
    import six
    import copy
    import cv2

    la = LateralNucleusRecurrent(la_size=1, la_map_size=(10, 10), la_in_size=3)
    la.reset()

    xs = [np.random.rand(3).reshape(1, 3)]
    y0 = copy.deepcopy(la.inference(xs, var=0.8, beta=0.5))
    y0 = y0.reshape(10, 10)
    la.update(lr=0.1, var=1.0)

    xs = [np.random.rand(3).reshape(1, 3)]
    y1 = copy.deepcopy(la.inference(xs, var=0.8, beta=0.5))
    y1 = y1.reshape(10, 10)
    la.update(lr=0.1, var=1.0)

    xs = [np.random.rand(3).reshape(1, 3)]
    y2 = copy.deepcopy(la.inference(xs, var=0.8, beta=0.5))
    y2 = y2.reshape(10, 10)
    la.update(lr=0.1, var=1.0)

    img0 = cv2.resize(y0, (200, 200), interpolation=cv2.INTER_NEAREST)
    img1 = cv2.resize(y1, (200, 200), interpolation=cv2.INTER_NEAREST)
    img2 = cv2.resize(y2, (200, 200), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('0', img0)
    cv2.imshow('1', img1)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
