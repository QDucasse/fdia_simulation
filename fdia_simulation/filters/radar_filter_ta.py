# -*- coding: utf-8 -*-
"""
Created on Tue Jul 02 09:17:23 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc               import x, y, z
from sympy                   import symbols, Matrix
from math                    import sqrt, atan2, exp
from scipy.linalg            import block_diag
from copy                    import deepcopy
from fdia_simulation.filters import RadarModel

class RadarFilterTA(RadarModel):
    r'''Implements a Kalman Filter state estimator for an aircraft-detecting
    radar. The model is assumed to have thrust acceleration.

    Notes
    -----
    The state transition function and matrix (f & F), the measurement function
    and matrix (h & H) and the process noise matrix (Q) are the main differences
    between the filter models.
    '''

    def compute_F(self,X):
        dt = self.dt
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
        self.F = F
        return self.F

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
