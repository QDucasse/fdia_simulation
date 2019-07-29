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
from fdia_simulation.filters import RadarFilterModel, MultipleRadarsFilterModel, MultipleFreqRadarsFilterModel

class RadarFilterTA(RadarFilterModel):
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

class MultipleRadarsFilterTA(RadarFilterTA,MultipleRadarsFilterModel):
    def __init__(self,*args,**kwargs):
        MultipleRadarsFilterModel.__init__(self,*args,**kwargs)

    def compute_F(self,X):
        return RadarFilterTA.compute_F(self,X)

    def compute_Q(self,q):
        return RadarFilterTA.compute_Q(self,q)

    def hx(self,X):
        return MultipleRadarsFilterModel.hx(self,X)

    def HJacob(self,X):
        return MultipleRadarsFilterModel.HJacob(self,X)


class MultipleFreqRadarsFilterTA(RadarFilterTA,MultipleFreqRadarsFilterModel):
    def __init__(self,*args,**kwargs):
        MultipleFreqRadarsFilterModel.__init__(self,*args,**kwargs)

    def compute_F(self,X):
        return RadarFilterTA.compute_F(self,X)

    def compute_Q(self,q):
        return RadarFilterTA.compute_Q(self,q)

    def hx(self,X,tag):
        return MultipleFreqRadarsFilterModel.hx(self,X,tag)

    def HJacob(self,X,tag):
        return MultipleFreqRadarsFilterModel.HJacob(self,X,tag)

    def update(self,labeled_z):
        MultipleFreqRadarsFilterModel.update(self,labeled_z)


if __name__ == "__main__":
    # Jacobian matrices determination using sympy
    vx,vy,vz,ax,ay,az = symbols('v_x, v_y, v_z, a_x, a_y, a_z')
    # =============== Constant Velocity/acceleration model =====================
    # For h
    hx = Matrix([[sympy.sqrt(x**2 + y**2 + z**2)],
                 [sympy.atan2(y,x)],
                 [sympy.atan2(z,sympy.sqrt(x**2 + y**2))]])
    hjac = hx.jacobian(Matrix([x, vx, ax, y, vy, ay, z, vz, az]))
    print(hjac)
    # For F, the matrices are not changing over time so no need for jacobians
