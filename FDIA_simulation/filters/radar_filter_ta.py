# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 09:17:23 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc       import x, y, z
from sympy           import symbols, Matrix
from math            import sqrt, atan2
from scipy.linalg    import block_diag
from fdia_simulation.filters.radar_filter_model import RadarModel

class RadarFilterTA(RadarModel):
    r'''Implements a Kalman Filter state estimator for an aircraft-detecting
    radar. The model is assumed to have thrust acceleration.
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
    def __init__(self,dim_x, dim_z, q,
                      x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                      vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                      ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                      dt = 1., std_r = 5., std_theta = 1., std_phi = 1.,
                      x_rad = 0., y_rad = 0., z_rad = 0.):

        edt = exp(dt)
        F = np.array([[1,edt-1, 0, 0,    0, 0, 0,    0, 0],
                      [0,  edt, 0, 0,    0, 0, 0,    0, 0],
                      [0,    0, 1, 0,    0, 0, 0,    0, 0],
                      [0,    0, 0, 1,edt-1, 0, 0,    0, 0],
                      [0,    0, 0, 0,  edt, 0, 0,    0, 0],
                      [0,    0, 0, 0,    0, 1, 0,    0, 0],
                      [0,    0, 0, 0,    0, 0, 1,edt-1, 0],
                      [0,    0, 0, 0,    0, 0, 0,  edt, 0],
                      [0,    0, 0, 0,    0, 0, 0,    0, 1]])

        RadarModel.__init__(self, dim_x = dim_x, dim_z = dim_z, F = F, q =q,
                            x0  = x0,  y0  = y0,  z0  = z0,
                            vx0 = vx0, vy0 = vy0, vz0 = vz0,
                            ax0 = ax0, ay0 = ay0, az0 = az0,
                            dt = dt, std_r = std_r, std_theta = std_theta, std_phi = std_phi,
                            x_rad = x_rad, y_rad = y_rad, z_rad = z_rad)

        self.x = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T

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


class TAMultipleRadars(RadarFilterTA):
    r'''Implements a filter model using multiple sensors and combining them
    through the measurement function and matrix.
    Parameters
    ----------
    radar_nb: int
        Number of radars.

    others, same as RadarFilterCA.
    '''
    def __init__(self,dim_x, dim_z, q,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 5., std_theta = 1., std_phi = 1.,
                       x_rad = 0., y_rad = 0., z_rad = 0.,radar_nb = 1):
        RadarFilterCA.__init__(self, dim_x, dim_z, q,
                           x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                           vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                           ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                           dt = 1., std_r = 5., std_theta = 1., std_phi = 1.,
                           x_rad = 0., y_rad = 0., z_rad = 0.)
        self.radar_nb = radar_nb
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
        Zpart = RadarFilterCA.hx(X)
        Z     = np.array([[]])
        for _ in range(self.radar_nb):
            Z = np.concatenate((Z,Zpart),axis=0)
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
        Hpart = RadarFilterCA.hx(X)
        H     = np.array([[]])
        for _ in range(self.radar_nb):
            H = np.concatenate((H,Hpart),axis=0)
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
    # For F, the matrices are not changing over time so no need for jacobians
