# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:29:27 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from filterpy.common import kinematic_kf


kf1 = kinematic_kf(2, 2)
kf2 = kinematic_kf(2, 2)
# do some settings of x, R, P etc. here, I'll just use the defaults
kf2.Q *= 0   # no prediction error in second filter

filters = [kf1, kf2]
mu = [0.5, 0.5]  # each filter is equally likely at the start
trans = np.array([[0.97, 0.03], [0.03, 0.97]])
imm = IMMEstimator(filters, mu, trans)
probs = []
for i in range(100):
 # make some noisy data
    x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
    y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
    z = np.array([[x], [y]])

    # perform predict/update cycle
    imm.predict()
    imm.update(z)
    probs.append(imm.mu)

probs = np.array(probs)
plt.plot(probs[:, 0])
plt.plot(probs[:, 1])
plt.show()
