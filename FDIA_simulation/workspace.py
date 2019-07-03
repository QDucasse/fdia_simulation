# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:29:27 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from filterpy.common import kinematic_kf


a = np.array([[1,2],
              [3,4]])
b = np.array([[5,6],
              [7,8]])

c = np.concatenate((a,b), axis = 1)
print(c)
