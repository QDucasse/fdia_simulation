# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:36 2019

@author: qde
"""

import sympy
import numpy as np
from math            import sqrt, atan2
from abc             import ABC,abstractmethod
from filterpy.kalman import ExtendedKalmanFilter


class RadarModel(ExtendedKalmanFilter,ABC):
    def __init__(self, dim_x, dim_z, F, q,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 5., std_theta = 0.01, std_phi = 0.01,
                       x_rad = 0., y_rad = 0., z_rad = 0.):

        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        self.F = F
        self.dt = dt
        self.x_rad = x_rad
        self.y_rad = y_rad
        self.z_rad = z_rad
        self.R = np.array([[std_r,0        ,       0],
                           [1    ,std_theta,       0],
                           [1    ,0        , std_phi]])
        self.compute_Q(q)

    @abstractmethod
    def HJacob(self,X):
        pass

    @abstractmethod
    def hx(self,X):
        pass

    @abstractmethod
    def compute_Q(self,q):
        pass

    def predict(self,u = 0):
        if u == None:
            u = 0
        ExtendedKalmanFilter.predict(self, u)

    def update(self,z, logs=False):
        z = np.reshape(z,(-2,1))
        H = self.HJacob(self.x)
        PHT = self.P@H.T
        self.S = H@PHT + self.R
        self.K = PHT@np.linalg.inv(self.S)
        hx = self.hx(self.x)
        self.y = np.subtract(z, hx)
        self.x = self.x + (self.K@self.y)

        if(logs):
            print('New Kalman gain: \n{0}\n'.format(self.K))
            print('Estimate: \n{0}\n'.format(hx))
            print('Innovation: \n{0}\n'.format(self.y))
            print('State space vector before correction: \n{0}\n'.format(self.x))
            print('Ponderated state: \n{0}\n'.format(self.K@self.y))
