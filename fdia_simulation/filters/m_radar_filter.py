# -*- coding: utf-8 -*-
"""
Created on Fri Jul 05 10:31:40 2019

@author: qde
"""

import numpy as np
from copy                    import deepcopy
from scipy.linalg            import block_diag
from filterpy.kalman         import ExtendedKalmanFilter
from fdia_simulation.filters import RadarModel,RadarFilterCV,RadarFilterCA,RadarFilterCT,RadarFilterTA


class MultipleRadarsFilter(RadarFilterCV,RadarFilterCA,RadarFilterCT,RadarFilterTA):
    r'''Implements a filter model using multiple sensors and combining them
    through the measurement function and matrix.
    '''
    def __init__(self,dim_x, dim_z, q, radars, model,
                 x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                 vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                 ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6):

        self.model = model
        model.__init__(self, dim_x, dim_z, q,
                       x0  = x0, y0  = y0, z0  = z0,
                       vx0 = vx0, vy0 = vy0, vz0 = vz0,
                       ax0 = ax0, ay0 = ay0, az0 = az0)
        self.radars          = radars
        self.radar_positions = [radar.get_position() for radar in radars]
        self.Rs              = [radar.R for radar in radars]
        self.R               = block_diag(*self.Rs)

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
            X_cur = deepcopy(X)
            X_cur[0,0] -= position[0]
            X_cur[3,0] -= position[1]
            X_cur[6,0] -= position[2]
            Z_part = self.model.hx(self,X_cur)
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
            X_cur = deepcopy(X)
            X_cur[0,0] -= position[0]
            X_cur[3,0] -= position[1]
            X_cur[6,0] -= position[2]
            H_part = self.model.HJacob(self,X_cur)
            H = np.concatenate((H,H_part),axis=0)
        return H

class MultipleFreqRadarsFilter(MultipleRadarsFilter):
    r'''Implements a filter model using multiple sensors with different data rates
    and combining them through the measurement function and matrix.
    '''
    def __init__(self,dim_x, dim_z, q, radars, model,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6):

        MultipleRadarsFilter.__init__(self, dim_x = dim_x, dim_z = dim_z, q = q,
                                     radars = radars, model = model,
                                     x0  = x0, y0  = y0, z0  = z0,
                                     vx0 = vx0, vy0 = vy0, vz0 = vz0,
                                     ax0 = ax0, ay0 = ay0, az0 = az0)
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
            X_cur = deepcopy(X)
            X_cur[0,0] -= position[0]
            X_cur[3,0] -= position[1]
            X_cur[6,0] -= position[2]
            if i == tag:
                Z_part = self.model.hx(self,X_cur)
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
            X_cur = deepcopy(X)
            X_cur[0,0] -= position[0]
            X_cur[3,0] -= position[1]
            X_cur[6,0] -= position[2]
            if i == tag: # If the radar if the one sending the measurement
                H_part = self.model.HJacob(self,X_cur)
            else:
                H_part = np.zeros((3,9))
            H = np.concatenate((H,H_part),axis=0)
        self.Hs.append(H)
        return H

    def update(self, labeled_z):
        '''
        Enhanced update method that needs to treat the labeled measurement and
        use the correct H matrix and function with the help of radar tag.
        Parameters
        ----------
        z: LabeledMeasurement
            The container of tag, time and measurement
        '''
        tag, t ,z = labeled_z.tag, labeled_z.time, np.array(labeled_z.value)
        self.dt      = t - self._last_t
        self._last_t = t
        self.compute_Q(self.q)
        self.compute_F(self.x)
        radars_nb = len(self.radars)
        z_input   = np.zeros((3*radars_nb,1))
        z = np.reshape(z,(-2,1))
        z_input[(3*tag):(3*tag+3),:] = z
        ExtendedKalmanFilter.update(self,z = z_input,
                                    HJacobian = self.HJacob, args = (tag),
                                    Hx = self.hx, hx_args = (tag))
