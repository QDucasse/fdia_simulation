# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:36 2019

@author: qde
"""

import sympy
import numpy as np
from math                            import sqrt, atan2
from copy                            import deepcopy
from abc                             import ABC,abstractmethod
from filterpy.kalman                 import ExtendedKalmanFilter
from fdia_simulation.models          import Radar
from fdia_simulation.fault_detectors import ChiSquareDetector


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

    def __init__(self, dim_x, dim_z, q, radar = None,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1.):

        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        self.dt    = dt
        if radar is None:
            radar = Radar(x=0,y=0,z=0)
        self.x_rad = radar.x
        self.y_rad = radar.y
        self.z_rad = radar.z
        self.R     = radar.R
        self.q     = q
        self.x = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T
        self.compute_Q(q)
        self.compute_F(self.x)
        self.detector = ChiSquareDetector()

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
    def compute_F(self,X):
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


    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        return np.subtract(z, self.HJacob(self.x)@self.x_prior)

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

    def update(self,z, HJacobian = None, Hx = None,
               args = (), hx_args = ()):
        '''
        Update step of the estimator.
        Parameters
        ----------
        z: numpy float array
            New measurement vector.

        logs: boolean
            Triggers the display of the in-state parameters.
        '''
        if HJacobian is None:
            HJacobian = self.HJacob
        if Hx is None:
            Hx = self.hx
        z = np.reshape(z,(-(self.dim_z-1),1))

        # Anomaly detector à mettre en place
        # Anomaly detection using the specified detector
        # res_detection = self.detector.review_measurement(z,self)
        # If res_detection = True => No problem in the measurement
        if True: #res_detection:
            ExtendedKalmanFilter.update(self,z = z, HJacobian = self.HJacob,
                                        Hx = self.hx, args = args, hx_args = hx_args)
        else:
            ExtendedKalmanFilter.update(self,z = None, HJacobian = self.HJacob,
                                        Hx = self.hx, args = args, hx_args = hx_args)
