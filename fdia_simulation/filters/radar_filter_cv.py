# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:40 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc               import x, y, z
from sympy                   import symbols, Matrix
from math                    import sqrt, atan2
from scipy.linalg            import block_diag
from copy                    import deepcopy
from fdia_simulation.filters import RadarModel, MultipleRadarsFilterModel

class RadarFilterCV(RadarModel):
    r'''Implements a Kalman Filter state estimator for an aircraft-detecting
    radar. The model is assumed to have constant velocity.

    Notes
    -----
    The state transition function and matrix (f & F), the measurement function
    and matrix (h & H) and the process noise matrix (Q) are the main differences
    between the filter models.
    '''

    def compute_F(self, X):
        dt = self.dt
        F = np.array([[1,dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1,dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1,dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
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
        Q_block = np.array([[dt**3/2, dt**2/2, 0],
                            [dt**2/2,      dt, 0],
                            [      0,       0, 0]])
        Q_block = q*Q_block
        self.Q = block_diag(Q_block, Q_block, Q_block)
        return self.Q

class MultipleRadarsFilterCV(RadarFilterCV,MultipleRadarsFilterModel):
    def compute_F(self,X):
        return RadarFilterCV.compute_F(self,X)

    def compute_Q(self,q):
        return RadarFilterCV.compute_Q(self,q)

    def hx(self,X):
        return MultipleRadarsFilterModel.hx(self,X)

    def HJacob(self,X):
        return MultipleRadarsFilterModel.HJacob(self,X)

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