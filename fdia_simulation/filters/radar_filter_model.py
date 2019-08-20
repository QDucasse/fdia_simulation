# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:36 2019

@author: qde
"""

import numpy as np
from math                              import sqrt, atan2
from abc                               import abstractmethod, ABC
from filterpy.kalman                   import ExtendedKalmanFilter
from filterpy.common                   import pretty_str
from fdia_simulation.models            import Radar


class RadarFilterModel(ExtendedKalmanFilter,ABC):
    '''Implements the basic utilities of radar filters and functions that will
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
        Initial positions of the airplane.

    vx0, vy0, vz0: floats
        Initial velocities of the airplane.

    ax0, ay0, az0: floats
        Initial accelerations of the airplane.

    dt: float
        Time step.

    std_r, std_theta, std_phi: floats
        Standard deviation of the measurement noise for the three values.

    x_rad, y_rad, z_rad: floats
        Radar position.
    '''

    def __init__(self, q, radar, dim_x = 9, dim_z = 3,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = None, detector = None):

        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        if dt is None:
            dt = Radar.DT_RADAR
        self.dt    = dt
        self.x_rad = radar.x
        self.y_rad = radar.y
        self.z_rad = radar.z
        self.R     = radar.R
        self.q     = q
        self.x = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T
        self.compute_Q(q)
        self.compute_F(self.x)
        self.detector = detector
        self.detection = False
        self.anomaly_counter = 0

    def HJacob(self,X):
        '''
        Computes the matrix H at a given point in time using the Jacobian of the
        function h.
        Parameters
        ----------
        X: numpy float array
            Space-state of the system.

        Returns
        -------
        H: numpy float array
            Jacobian of the h function applied to the state-space X at current
            time.
        '''
        x = X[0,0] - self.x_rad
        y = X[3,0] - self.y_rad
        z = X[6,0] - self.z_rad
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), 0, 0, y/sqrt(x**2 + y**2 + z**2), 0, 0, z/sqrt(x**2 + y**2 + z**2),0 ,0],
                      [-y/(x**2 + y**2), 0, 0, x/(x**2 + y**2), 0, 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0]])
        return H

    def hx(self,X):
        '''
        Computes the h measurement function (when applied to the state space,
        should output the measurements).
        Parameters
        ----------
        X: numpy float array
            Space-state of the system.

        Returns
        -------
        Z_k: numpy float array
            Kth measurement as outputed by the measurement function.
        '''
        # State space vector
        x = X[0,0] - self.x_rad
        y = X[3,0] - self.y_rad
        z = X[6,0] - self.z_rad
        # Measurements
        r     = sqrt(x**2 + y**2 + z**2)
        theta = atan2(y,x)
        phi   = atan2(z,sqrt(x**2 + y**2))
        # Measurement vector
        Z_k = np.array([[r,theta,phi]]).T
        return Z_k

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
        if u is None:
            u = 0
        ExtendedKalmanFilter.predict(self, u = u)

    def activate_detection(self):
        '''
        Switches the detection boolean triggering the anomaly detection on
        measurements.
        '''
        self.detection = True

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
        # Anomaly detection using the specified detector
        if not(self.detector is None) and self.detection:
            res_detection = self.detector.review_measurement(z,self)
            if not(res_detection):
                z = None
                self.anomaly_counter += 1
        self.detection = False
        # If res_detection = True => No problem in the measurement
        ExtendedKalmanFilter.update(self,z = z, HJacobian = HJacobian,
                                    Hx = Hx, args = args, hx_args = hx_args)
    # def __repr__(self):
    #     return '\n'.join([
    #         'RadarFilter object',
    #         pretty_str('Name', type(self).__name__[-2:]),
    #         pretty_str('Time unit', self.dt),
    #         pretty_str('Radar position', [self.x_rad,self.y_rad,self.z_rad]),
    #         pretty_str('Measurement noise', self.R),
    #         pretty_str('Process noise', self.Q),
    #         pretty_str('Transition matrix', self.F)])
