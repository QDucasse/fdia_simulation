# -*- coding: utf-8 -*-
"""
Created on Fri Jul 05 10:31:40 2019

@author: qde
"""

import numpy as np
from copy import deepcopy
# from fdia_simulation.filters.radar_filter_model import RadarModel
from scipy.linalg import block_diag
from fdia_simulation.filters.radar_filter_cv import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca import RadarFilterCA
from fdia_simulation.filters.radar_filter_ct import RadarFilterCT
from fdia_simulation.filters.radar_filter_ta import RadarFilterTA

class MultipleRadarsFilter(RadarFilterCV,RadarFilterCA,RadarFilterCT,RadarFilterTA):
    r'''Implements a filter model using multiple sensors and combining them
    through the measurement function and matrix.
    '''
    def __init__(self,dim_x, dim_z, q, radars, model,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1.):

        model.__init__(self, dim_x, dim_z, q,
                       x0  = x0, y0  = y0, z0  = z0,
                       vx0 = vx0, vy0 = vy0, vz0 = vz0,
                       ax0 = ax0, ay0 = ay0, az0 = az0,
                       dt = dt)
        self.model           = model
        self.radars          = radars
        self.radar_positions = [radar.get_position() for radar in radars]
        self.Rs              = [radar.R for radar in radars]
        self.R               = block_diag(*self.Rs)

    def hx(self,X):
        '''
        Computes the h function: Concatenation of radar_nb h functions.
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
        radar_nb Jacobians.
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
