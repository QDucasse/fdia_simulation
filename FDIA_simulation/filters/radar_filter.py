# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:46:27 2019

@author: qde
"""

import numpy as np
from math import sqrt
from filterpy.kalman import ExtendedKalmanFilter




# Other model possibility
    # Following [ARIF2017] paper
    # X_next      = np.zeros((6,1))
    # X_next[0,0] = x / sqrt(x**2 + y**2 + z**2) - y - x * z / sqrt(x**2 + y**2)
    # X_next[1,0] = y / sqrt(x**2 + y**2 + z**2) - y - y * z / sqrt(x**2 + y**2)
    # X_next[2,0] = z / sqrt(x**2 + y**2 + z**2) + sqrt(x**2 + y**2)
    # X_next[3,0] = -2*y / sqrt(x**2 + y**2 + z**2) - 2*x*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*x + 2*y*z / sqrt(x**2 + y**2)
    # X_next[4,0] = -2*x / sqrt(x**2 + y**2 + z**2) - 2*y*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*y + 2*x*z / sqrt(x**2 + y**2)
    # X_next[5,0] = 2 * sqrt(x**2 + y**2) / sqrt(x**2 + y**2 + z**2)



class RadarFilter(ExtendedKalmanFilter):

    def __init__(self, X, std_r = 1., std_theta = 0.01, std_phi = 0.01):
        super().__init__(dim_x=6,dim_z=3,)
        self.F = np.array([[1, 0, 0,dt, 0, 0],
                           [0, 1, 0, 0,dt, 0],
                           [0, 0, 1, 0, 0,dt],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.R = np.array([[std_r,0        ,       0],
                           [1    ,std_theta,       0],
                           [1    ,0        , std_phi]])

    def Hjacob(X):
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), y/sqrt(x**2 + y**2 + z**2), z/sqrt(x**2 + y**2 + z**2), 0, 0, 0],
                      [-y/(x**2 + y**2), x/(x**2 + y**2), 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0, 0]])
        return H

    def h(X):
        # State space vector
        x = X[0,0]
        y = X[1,0]
        z = X[2,0]
        # Measurements
        r     = sqrt(x**2 + y**2 + z**2)
        theta = atan2(y,x)
        phi   = atan2(z,sqrt(x**2 + y**2))
        # Measurement vector
        Z_k = np.array([[r,theta,phi]]).T
        return Z_k



if __name__ == "__main__":
    pass
