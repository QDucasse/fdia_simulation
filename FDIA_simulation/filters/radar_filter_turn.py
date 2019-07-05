# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:32 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc       import x, y, z
from sympy           import symbols, Matrix
from math            import sqrt, atan2, cos, sin
from scipy.linalg    import block_diag
from copy            import deepcopy
from fdia_simulation.filters.radar_filter_model import RadarModel

class RadarFilterTurn(RadarModel):
    r'''Implements a Kalman Filter state estimator for an aircraft-detecting
    radar. The model is assumed to have constant velocity.
    Parameters
    ---------
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

    Notes
    -----
    The state transition function and matrix (f & F), the measurement function
    and matrix (h & H) and the process noise matrix (Q) are the main differences
    between the filter models.
    '''
    def __init__(self, dim_x, dim_z, q, radar = None,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1.):

        self.x = np.array([[x0,y0,z0,vx0,vy0,vz0,ax0,ay0,az0]]).T
        F = np.eye(9)
        RadarModel.__init__(self, dim_x = dim_x, dim_z = dim_z,
                            F = F, q =q, radar = radar,
                            x0  = x0,  y0  = y0,  z0  = z0,
                            vx0 = vx0, vy0 = vy0, vz0 = vz0,
                            ax0 = ax0, ay0 = ay0, az0 = az0,
                            dt = dt)
        F = self.compute_F(self.x)


    def compute_Q(self,q):
        '''
        Computes process noise.
        Parameters
        ----------
        q: float
            Input process noise.
        Returns
        -------
        Q: numpy float array
            The process noise matrix.
        '''
        dt = self.dt
        Q_block = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0,dt]])
        Q_block = q*Q_block
        self.Q = block_diag(Q_block, Q_block, Q_block)
        return self.Q

    # def FJacob(self, X):
    #     x   = X[0,0]
    #     v_x = X[1,0]
    #     a_x = X[2,0]
    #     y   = X[3,0]
    #     v_y = X[4,0]
    #     a_y = X[5,0]
    #     z   = X[6,0]
    #     v_z = X[7,0]
    #     a_z = X[8,0]
    #     F = np.array([[1, 2*a_x*v_x*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_x*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_x**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_x**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2), -2*a_x**2*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_x**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_x*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*v_x*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + (1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2), 0, 2*a_x*v_y*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_x*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_x*a_y*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_x*a_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_y*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y*v_x*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2), 0, 2*a_x*v_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_x*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_x*a_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_x*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_z*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_z*v_x*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2)], [0, -1.0*a_x*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_x*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_x**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)), 1.0*a_x**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_x*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2), 0, -1.0*a_x*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_x*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_x*a_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*a_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_y*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, -1.0*a_x*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_x*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_x*a_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_z*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2))], [0, 1.0*a_x*v_x*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_x**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_x**2*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2 - sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(v_x**2 + v_y**2 + v_z**2), -1.0*a_x**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_x*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_x*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_x*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_x*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_x*v_y*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_x*a_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_y*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_y*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_x*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_x*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_x*v_z*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_x*a_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_z*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_z*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2))], [0, 2*a_y*v_x*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_y*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_x*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_x*a_y*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_x*a_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_x*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*v_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2), 1, 2*a_y*v_y*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_y*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_y**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_y**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2), -2*a_y**2*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_y**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_y*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y*v_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + (1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2), 0, 2*a_y*v_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_y*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_y*a_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_y*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_z*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_z*v_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2)], [0, -1.0*a_y*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_y*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_x*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_x*a_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*a_y*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, -1.0*a_y*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_y*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_y**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)), 1.0*a_y**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_y*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2), 0, -1.0*a_y*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_y*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_y*a_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_z*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2))], [0, 1.0*a_y*v_x*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_x*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_x*v_y*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_x*a_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_x*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_x*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_y*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_y**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_y**2*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2 - sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(v_x**2 + v_y**2 + v_z**2), -1.0*a_y**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_y*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_y*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_y*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_y*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_y*v_z*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_y*a_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_z*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_z*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2))], [0, 2*a_z*v_x*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_z*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_x*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_x*a_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_x*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_x*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*v_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2), 0, 2*a_z*v_y*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_z*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_y*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), -2*a_y*a_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_y*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_y*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y*v_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2), 1, 2*a_z*v_z*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))/(a_x**2 + a_y**2 + a_z**2) - 1.0*a_z*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*v_z**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + v_z**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2), -2*a_z**2*(1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)**2 + 1.0*a_z**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + 1.0*a_z*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_z*v_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) + (1 - cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)))*(v_x**2 + v_y**2 + v_z**2)/(a_x**2 + a_y**2 + a_z**2)], [0, -1.0*a_z*v_x*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_z*v_x*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_x*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_x*a_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_x*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, -1.0*a_z*v_y*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_z*v_y*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_y*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2), 1.0*a_y*a_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_y*a_z*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, -1.0*a_z*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) + a_z*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + 1.0*v_z**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2)), 1.0*a_z**2*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2) - a_z**2*sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(a_x**2 + a_y**2 + a_z**2)**(3/2) - 1.0*a_z*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + sqrt(v_x**2 + v_y**2 + v_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(a_x**2 + a_y**2 + a_z**2)], [0, 1.0*a_z*v_x*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_x*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_x*v_z*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_x*a_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_x*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_x*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_z*v_y*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_y*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_y*v_z*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2, -1.0*a_y*a_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_y*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_y*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)), 0, 1.0*a_z*v_z*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + v_z**2*sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**(3/2) + 1.0*v_z**2*(a_x**2 + a_y**2 + a_z**2)*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2)**2 - sqrt(a_x**2 + a_y**2 + a_z**2)*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/sqrt(v_x**2 + v_y**2 + v_z**2), -1.0*a_z**2*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) - 1.0*a_z*v_z*cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(v_x**2 + v_y**2 + v_z**2) - a_z*v_z*sin(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))/(sqrt(a_x**2 + a_y**2 + a_z**2)*sqrt(v_x**2 + v_y**2 + v_z**2)) + cos(1.0*sqrt(a_x**2 + a_y**2 + a_z**2)/sqrt(v_x**2 + v_y**2 + v_z**2))]])
    #     return F
    #
    # def fx(self, X):
    #     x  = X[0,0] - self.x_rad
    #     vx = X[1,0]
    #     ax = X[2,0]
    #     y  = X[3,0] - self.y_rad
    #     vy = X[4,0]
    #     ay = X[5,0]
    #     z  = X[6,0] - self.z_rad
    #     vz = X[7,0]
    #     az = X[8,0]
    #     dt = self.dt
    #     omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
    #     x_hat  = x + sin(omega*dt)/omega * vx + (1 - cos(omega*dt))/omega**2* ax
    #     vx_hat =     cos(omega*dt)       * vx + sin(omega*dt)/omega         * ax
    #     ax_hat =     -omega*sin(omega*dt)* vx + cos(omega*dt)               * ax
    #     y_hat  = y + sin(omega*dt)/omega * vy + (1 - cos(omega*dt))/omega**2* ay
    #     vy_hat =     cos(omega*dt)       * vy + sin(omega*dt)/omega         * ay
    #     ay_hat =     -omega*sin(omega*dt)* vy + cos(omega*dt)               * ay
    #     z_hat  = z + sin(omega*dt)/omega * vz + (1 - cos(omega*dt))/omega**2* az
    #     vz_hat =     cos(omega*dt)       * vz + sin(omega*dt)/omega         * az
    #     az_hat =     -omega*sin(omega*dt)* vz + cos(omega*dt)               * az
    #     return np.array([[x_hat,vx_hat,ax_hat,y_hat,vy_hat,ay_hat,z_hat,vz_hat,az_hat]])

    def compute_F(self, X):
        vx = X[1,0]
        ax = X[2,0]
        vy = X[4,0]
        ay = X[5,0]
        vz = X[7,0]
        az = X[8,0]
        dt = self.dt
        omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
        F_block = np.array([[1,  sin(omega*dt)/omega, (1 - cos(omega*dt))/omega**2],
                            [0,        cos(omega*dt),          sin(omega*dt)/omega],
                            [0, -omega*sin(omega*dt),                cos(omega*dt)]])
        self.F = block_diag(F_block,F_block,F_block)
        return self.F


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

    def predict_x(self, u=0):
        RadarModel.predict_x(self, u)
        self.compute_F(self.x)



class TurnMultipleRadars(RadarFilterTurn):
    r'''Implements a filter model using multiple sensors and combining them
    through the measurement function and matrix.
    Parameters
    ----------
    radars: Radar iterable
        List of radars observing the airplane.

    same as RadarFilterTurn.
    '''
    def __init__(self,dim_x, dim_z, q, radars,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1.):
        RadarFilterTurn.__init__(self, dim_x, dim_z, q,
                                 x0  = x0, y0  = y0, z0  = z0,
                                 vx0 = vx0, vy0 = vy0, vz0 = vz0,
                                 ax0 = ax0, ay0 = ay0, az0 = az0,
                                 dt = dt)
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
            Z_part = RadarFilterTurn.hx(self,X_cur)
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
            H_part = RadarFilterTurn.HJacob(self,X_cur)
            H = np.concatenate((H,H_part),axis=0)
        return H


if __name__ == "__main__":
    # Jacobian matrices determination using sympy
    vx,vy,vz,ax,ay,az = symbols('v_x, v_y, v_z, a_x, a_y, a_z')
    # =============== Constant Velocity/acceleration model =====================
    # For h
    hx = Matrix([[sympy.sqrt(x**2 + y**2 + z**2)],
                 [sympy.atan2(y,x)],
                 [sympy.atan2(z,sympy.sqrt(x**2 + y**2))]])
    hjac = hx.jacobian(Matrix([x, vx, ax, y, vy, ay, z, vz, az]))
    #===========================================================================
    # =========================== Matrices Display =============================
    # print("Jacobian of the measurement function: \n{0}\n".format(hjac))
