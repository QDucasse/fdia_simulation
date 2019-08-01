# -*- coding: utf-8 -*-
"""
Created on Fri Jul 05 10:31:40 2019

@author: qde
"""

import numpy as np
from math                    import sqrt, atan2
# from copy                    import deepcopy
from scipy.linalg            import block_diag
from filterpy.kalman         import ExtendedKalmanFilter
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarFilterModel


class MultipleRadarsFilterModel(RadarFilterModel):
    r'''Implements a filter model using multiple sensors and combining them
    through the measurement function and matrix.
    '''
    def __init__(self, q, radars, dim_x = 9, dim_z = None,
                 x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                 vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                 ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                 dt = None, detector = None):

        if dim_z is None:
            dim_z = 3*len(radars)
        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        if dt is None:
            dt = Radar.DT_RADAR
        self.dt              = dt
        self.radars          = radars
        self.radar_positions = [radar.get_position() for radar in radars]
        self.Rs              = [radar.R for radar in radars]
        self.R               = block_diag(*self.Rs)
        self.q               = q
        self.x               = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T
        self.compute_Q(q)
        self.compute_F(self.x)
        self.detector = detector


    def hx(self,X):
        '''
        Computes h, the measurement function: Concatenation of n (number of radars)
        h functions.
        Parameters
        ----------
        X: numpy float array
            State space vector.

        Returns
        -------
        Z: numpy float array
            Concatenated measurement function output.

        Notes
        -----
        The result is obtained by vertically concatenating the measurement
        function of one radar for each of them.
        '''

        Z = np.reshape(np.array([[]]),(0,1))
        for position in self.radar_positions:
            # X_cur = deepcopy(X)
            # X_cur[0,0] -= position[0]
            # X_cur[3,0] -= position[1]
            # X_cur[6,0] -= position[2]
            # Z_part = RadarFilterModel.hx(self,X_cur)
            x = X[0,0] - position[0]
            y = X[3,0] - position[1]
            z = X[6,0] - position[2]
            r     = sqrt(x**2 + y**2 + z**2)
            theta = atan2(y,x)
            phi   = atan2(z,sqrt(x**2 + y**2))
            Z_part = np.array([[r,theta,phi]]).T
            Z = np.concatenate((Z,Z_part),axis=0)
        return Z

    def HJacob(self,X):
        '''
        Computes H, the Jacobian of the h measurement function. Concatenation of
        n (number of radars) Jacobians.
        Parameters
        ----------
        X: numpy float array
            State space vector.

        Returns
        -------
        H: numpy float array
            Concatenated measurement function Jacobian.
        '''
        H = np.reshape(np.array([[]]),(0,9))
        for position in self.radar_positions:
            # X_cur = deepcopy(X)
            # X_cur[0,0] -= position[0]
            # X_cur[3,0] -= position[1]
            # X_cur[6,0] -= position[2]
            # H_part = RadarFilterModel.HJacob(self,X_cur)
            x = X[0,0] - position[0]
            y = X[3,0] - position[1]
            z = X[6,0] - position[2]
            H_part = np.array([[x/sqrt(x**2 + y**2 + z**2), 0, 0, y/sqrt(x**2 + y**2 + z**2), 0, 0, z/sqrt(x**2 + y**2 + z**2),0 ,0],
                               [-y/(x**2 + y**2), 0, 0, x/(x**2 + y**2), 0, 0, 0, 0, 0],
                               [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0]])
            H = np.concatenate((H,H_part),axis=0)
        return H

class MultiplePeriodRadarsFilterModel(MultipleRadarsFilterModel):
    r'''Implements a filter model using multiple sensors with different data rates
    and combining them through the measurement function and matrix.
    '''
    def __init__(self,*args,**kwargs):
        MultipleRadarsFilterModel.__init__(self,*args,**kwargs)
        self._last_t = 0
        self._tag_radars()
        self.Hs = []
        self.Zs = []

    def _tag_radars(self):
        '''
        Attributes tags to radars. Tags are the positions of each radar in the
        radars list attribute.
        '''
        for i,radar in enumerate(self.radars):
            radar.tag = i

    def hx(self, X, tag):
        '''
        Computes h, the measurement function: Concatenation of n (number of radars)
        h functions.
        Parameters
        ----------
        X: numpy float array
            State space vector.

        tag: int
            Current radar the filter is getting the measurement from.

        Returns
        -------
        Z: numpy float array
            Concatenated measurement function output.

        Notes
        -----
        The result is obtained by vertically concatenating the measurement
        function result of one radar for each of them. Each of them are null except
        for the tagged radar.
        '''

        Z = np.reshape(np.array([[]]),(0,1))
        for i,position in enumerate(self.radar_positions):
            # X_cur = deepcopy(X)
            # X_cur[0,0] -= position[0]
            # X_cur[3,0] -= position[1]
            # X_cur[6,0] -= position[2]
            x = X[0,0] - position[0]
            y = X[3,0] - position[1]
            z = X[6,0] - position[2]
            if i == tag:
                # Z_part = RadarFilterModel.hx(self,X_cur)
                r     = sqrt(x**2 + y**2 + z**2)
                theta = atan2(y,x)
                phi   = atan2(z,sqrt(x**2 + y**2))
                Z_part = np.array([[r,theta,phi]]).T
            else:
                Z_part = np.zeros((3,1))
            Z = np.concatenate((Z,Z_part),axis=0)
        self.Zs.append(Z)
        return Z

    def HJacob(self,X, tag):
        '''
        Computes H, the Jacobian of the h measurement function. Concatenation of
        n (number of radars) Jacobians.
        Parameters
        ----------
        X: numpy float array
            State space vector.

        Returns
        -------
        H: numpy float array
            Concatenated measurement function Jacobian.
        '''
        H = np.reshape(np.array([[]]),(0,9))
        for i,position in enumerate(self.radar_positions):
            # X_cur = deepcopy(X)
            # X_cur[0,0] -= position[0]
            # X_cur[3,0] -= position[1]
            # X_cur[6,0] -= position[2]
            x = X[0,0] - position[0]
            y = X[3,0] - position[1]
            z = X[6,0] - position[2]
            if i == tag: # If the radar if the one sending the measurement
                # H_part = RadarFilterModel.HJacob(self,X_cur)
                H_part = np.array([[x/sqrt(x**2 + y**2 + z**2), 0, 0, y/sqrt(x**2 + y**2 + z**2), 0, 0, z/sqrt(x**2 + y**2 + z**2),0 ,0],
                                   [-y/(x**2 + y**2), 0, 0, x/(x**2 + y**2), 0, 0, 0, 0, 0],
                                   [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0]])
            else:
                H_part = np.zeros((3,9))
            H = np.concatenate((H,H_part),axis=0)
        self.Hs.append(H)
        return H

    def residual_of(self, tag, z):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        z_input = self.gen_complete_measurement(tag,z)
        return np.subtract(z_input, self.HJacob(self.x,tag = tag)@self.x_prior)


    def gen_complete_measurement(self,tag,z):
        '''
        Generates the whole measurement from one labeled measurement.
        '''
        radars_nb = len(self.radars)
        z_input   = np.zeros((3*radars_nb,1))
        z         = np.reshape(z,(-2,1))
        z_input[(3*tag):(3*tag+3),:] = z
        return z_input

    def update(self, labeled_z):
        '''
        Enhanced update method that needs to treat the labeled measurement and
        use the correct H matrix and function with the help of radar tag.
        Parameters
        ----------
        z: LabeledMeasurement
            The container of tag, time and measurement
        '''
        tag, t, z = labeled_z.tag, labeled_z.time, np.array(labeled_z.value)
        self.dt      = t - self._last_t
        self._last_t = t
        self.compute_Q(self.q)
        self.compute_F(self.x)
        radars_nb = len(self.radars)
        z_input = self.gen_complete_measurement(tag = tag, z = z)
        ExtendedKalmanFilter.update(self,z = z_input,
                                    HJacobian = self.HJacob, args = (tag),
                                    Hx = self.hx, hx_args = (tag))
