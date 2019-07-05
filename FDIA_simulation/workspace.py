# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:29:27 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from filterpy.common import kinematic_kf



class A(object):
    def bloub(self):
        return 'bloub'

class B(object):
    def __init__(self,a):
        print(a.bloub(self))

if __name__ == "__main__":
    b = B(A)
