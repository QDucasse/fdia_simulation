# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:32 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc               import x, y, z
from sympy                   import symbols, Matrix
from math                    import sqrt, atan2, cos, sin
from scipy.linalg            import block_diag
from copy                    import deepcopy
from fdia_simulation.filters import RadarModel,MultipleRadarsFilterModel

class RadarFilterCT(RadarModel):
    r'''Implements a Kalman Filter state estimator for an aircraft-detecting
    radar. The model is assumed to have constant velocity.

    Notes
    -----
    The state transition function and matrix (f & F), the measurement function
    and matrix (h & H) and the process noise matrix (Q) are the main differences
    between the filter models.
    '''

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

    def predict_x(self, u=0):
        RadarModel.predict_x(self, u)
        self.compute_F(self.x)


class MultipleRadarsFilterCT(RadarFilterCT,MultipleRadarsFilterModel):
    def compute_F(self,X):
        return RadarFilterCT.compute_F(self,X)

    def compute_Q(self,q):
        return RadarFilterCT.compute_Q(self,q)

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
    #===========================================================================
    # =========================== Matrices Display =============================
    # print("Jacobian of the measurement function: \n{0}\n".format(hjac))
