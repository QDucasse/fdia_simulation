# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:50:40 2019

@author: qde
"""
import sympy
import numpy as np
from sympy.abc       import x, y, z
from sympy           import symbols, Matrix
from math            import sqrt, atan2
from scipy.linalg    import block_diag
from fdia_simulation.filters.radar_filter_model import RadarModel


class RadarFilterCV(RadarModel):

    def __init__(self, dim_x, dim_z, q,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 50., std_theta = 0.05, std_phi = 0.05,
                       x_rad = 0., y_rad = 0., z_rad = 0.):

        F = np.array([[1,dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1,dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1,dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        RadarModel.__init__(self, dim_x = dim_x, dim_z = dim_x, F = F, q = q,
                            x0  = x0,  y0  = y0,  z0  = z0,
                            vx0 = vx0, vy0 = vy0, vz0 = vz0,
                            dt = dt, std_r = std_r, std_theta = std_theta, std_phi = std_phi,
                            x_rad = x_rad, y_rad = y_rad, z_rad = z_rad)

        self.x = np.array([[x0,vx0,ax0,y0,vy0,ay0,z0,vz0,az0]]).T


    def compute_Q(self,q):
        dt = self.dt
        Q_block = np.array([[dt**3/2, dt**2/2, 0],
                            [dt**2/2,      dt, 0],
                            [      0,       0, 0]])
        Q_block = q*Q_block
        self.Q = block_diag(Q_block, Q_block, Q_block)
        return self.Q

    def HJacob(self,X):
        x = X[0,0] - self.x_rad
        y = X[3,0] - self.y_rad
        z = X[6,0] - self.z_rad
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), 0, 0, y/sqrt(x**2 + y**2 + z**2), 0, 0, z/sqrt(x**2 + y**2 + z**2),0 ,0],
                      [-y/(x**2 + y**2), 0, 0, x/(x**2 + y**2), 0, 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0]])
        return H

    def hx(self,X):
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
