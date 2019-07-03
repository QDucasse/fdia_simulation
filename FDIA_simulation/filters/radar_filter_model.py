# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:36 2019

@author: qde
"""

import sympy
import numpy as np
from math            import sqrt, atan2
from copy            import deepcopy
from abc             import ABC,abstractmethod
from filterpy.kalman import ExtendedKalmanFilter


class RadarModel(ExtendedKalmanFilter,ABC):
    r'''Implements the basic utilities of radar filters and functions that will
    need to be overiden by subclasses.
    Parameters
    ---------
    dim_x: int
        Space-state vector dimension.

    dim_y: int
        Measurements vector dimension.

    F: numpy float matrix
        State transition matrix.

    q: float
        Process noise input.

    x0, y0, z0: floats
        Initial positions of the aircraft.

    vx0, vy0, vz0: floats
        Initial velocities of the aircraft.

    ax0, ay0, az0: floats
        Initial accelerations of the aircraft.

    dt: float
        Time step.

    std_r, std_theta, std_phi: floats
        Standard deviation of the measurement noise for the three values.

    x_rad, y_rad, z_rad: floats
        Radar position.
    '''
    def __init__(self, dim_x, dim_z, F, q,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 5., std_theta = 1., std_phi = 1.,
                       x_rad = 0., y_rad = 0., z_rad = 0.):

        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        self.F = F
        self.dt = dt
        self.x_rad = x_rad
        self.y_rad = y_rad
        self.z_rad = z_rad
        self.R = np.array([[std_r,0        ,       0],
                           [0    ,std_theta,       0],
                           [0    ,0        , std_phi]])
        self.compute_Q(q)
        self.x = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T

    @abstractmethod
    def HJacob(self,X):
        '''
        Computes the output of the Jacobian matrix of the measurement function
        for a given space state.
        Parameters
        ----------
        X: numpy float array
            State-space vector
        '''
        pass

    @abstractmethod
    def hx(self,X):
        '''
        Computes the output of the measurement function for a given space state.
        Parameters
        ----------
        X: numpy float array
            State-space vector
        '''
        pass

    @abstractmethod
    def compute_Q(self,q):
        '''
        Computes the process noise matrix of the estimator.
        Parameters
        ----------
        q: float
            Process noise input.

        Returns
        -------
        Q: numpy float array
            Process noise matrix
        '''
        pass

    def predict(self,u = 0):
        '''
        Prediction step of the estimator.
        Parameters
        ----------
        u: float
            Input of the system.

        Notes
        -----
        This is a correction of the behavior of IMM and ExtendedKalman Filter not
        working correctly together.
        '''
        if u == None:
            u = 0
        ExtendedKalmanFilter.predict(self, u = u)

    def update(self,z, logs=False):
        '''
        Update step of the estimator.
        Parameters
        ----------
        z: numpy float array
            New measurement vector.

        logs: boolean
            Triggers the display of the in-state parameters.
        '''
        z = np.reshape(z,(-2,1))
        ExtendedKalmanFilter.update(self,z = z, HJacobian = self.HJacob, Hx = self.hx)

        # H = self.HJacob(self.x)
        # PHT = self.P@H.T
        # self.S = H@PHT + self.R
        # self.K = PHT@np.linalg.inv(self.S)
        # hx = self.hx(self.x)
        # self.y = np.subtract(z, hx)
        # self.x = self.x + (self.K@self.y)

        if(logs):
            print('New Kalman gain: \n{0}\n'.format(self.K))
            print('Estimate: \n{0}\n'.format(hx))
            print('Innovation: \n{0}\n'.format(self.y))
            print('State space vector after correction: \n{0}\n'.format(self.x))
            print('KH: \n{0}\n'.format(self.K@H))
            print('I-KH: \n{0}\n'.format(I_KH))
            print('P: \n{0}\n'.format(self.P))
            print('Ponderated state: \n{0}\n'.format(self.K@self.y))
