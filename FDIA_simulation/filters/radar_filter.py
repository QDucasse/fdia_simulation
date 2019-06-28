# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:46:27 2019

@author: qde
"""

import sympy
import numpy as np
from sympy.abc       import x, y, z
from sympy           import symbols, Matrix
from math            import sqrt, atan2
from abc             import ABC,abstractmethod
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Other model possibility
    # Following [ARIF2017] paper
    # X_next      = np.zeros((6,1))
    # X_next[0,0] = x / sqrt(x**2 + y**2 + z**2) - y - x * z / sqrt(x**2 + y**2)
    # X_next[1,0] = y / sqrt(x**2 + y**2 + z**2) - y - y * z / sqrt(x**2 + y**2)
    # X_next[2,0] = z / sqrt(x**2 + y**2 + z**2) + sqrt(x**2 + y**2)
    # X_next[3,0] = -2*y / sqrt(x**2 + y**2 + z**2) - 2*x*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*x + 2*y*z / sqrt(x**2 + y**2)
    # X_next[4,0] = -2*x / sqrt(x**2 + y**2 + z**2) - 2*y*z / sqrt(x**2 + y**2 + z**2) / sqrt(x**2 + y**2) \
    #               -2*y + 2*x*z / sqrt(x**2 + y**2)
    # X_next[5,0] = 2 * sqrt(x**2 + y**2) / sqrt(x**2 + y**2 + z**2)

class RadarModel(ExtendedKalmanFilter,ABC):
    def __init__(self, dim_x, dim_z, F,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 10., std_theta = 0.1, std_phi = 0.1,
                       x_rad = 0., y_rad = 0., z_rad = 0.):

        ExtendedKalmanFilter.__init__(self, dim_x = dim_x, dim_z = dim_z)
        self.F = F
        self.dt = dt
        self.x_rad = x_rad
        self.y_rad = y_rad
        self.z_rad = z_rad
        self.R = np.array([[std_r,0        ,       0],
                           [1    ,std_theta,       0],
                           [1    ,0        , std_phi]])

    @abstractmethod
    def HJacob(self,X):
        pass

    @abstractmethod
    def hx(self,X):
        pass

    def predict(self,u = 0):
        if u == None:
            u = 0
        super().predict(u)

    def update(self,z, logs=False):
        z = np.reshape(z,(-2,1))
        H = self.HJacob(self.x)
        PHT = self.P@H.T
        self.S = H@PHT + self.R
        self.K = PHT@np.linalg.inv(self.S)
        hx = self.hx(self.x)
        self.y = np.subtract(z, hx)
        self.x = self.x + (self.K@self.y)

        if(logs):
            print('New Kalman gain: \n{0}\n'.format(self.K))
            print('Estimate: \n{0}\n'.format(hx))
            print('Innovation: \n{0}\n'.format(self.y))
            print('State space vector before correction: \n{0}\n'.format(self.x))
            print('Ponderated state: \n{0}\n'.format(self.K@self.y))



class RadarFilterCV(RadarModel):

    def __init__(self, dim_x, dim_z,
                       x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                       vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                       ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                       dt = 1., std_r = 1., std_theta = 0.05, std_phi = 0.05,
                       x_rad = 0., y_rad = 0., z_rad = 0.):

        F = np.array([[1, 0, 0,dt, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0,dt, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0,dt, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        RadarModel.__init__(self, dim_x = dim_x, dim_z = dim_x, F = F,
                            x0  = x0,  y0  = y0,  z0  = z0,
                            vx0 = vx0, vy0 = vy0, vz0 = vz0,
                            dt = dt, std_r = std_r, std_theta = std_theta, std_phi = std_phi,
                            x_rad = x_rad, y_rad = y_rad, z_rad = z_rad)

        self.x = np.array([[x0,y0,z0,vx0,vy0,vz0,ax0,ay0,az0]]).T

        # self.Q = np.zeros((9,9))
        # self.Q[0:3,0:3] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.1)
        # self.Q[3:6,3:6] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.1)
        # self.Q[6:9,6:9] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.1)


    def HJacob(self,X):
        x = X[0,0] - self.x_rad
        y = X[1,0] - self.y_rad
        z = X[2,0] - self.z_rad
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), y/sqrt(x**2 + y**2 + z**2), z/sqrt(x**2 + y**2 + z**2), 0, 0, 0, 0, 0, 0],
                      [-y/(x**2 + y**2), x/(x**2 + y**2), 0, 0, 0, 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0, 0, 0, 0, 0]])
        return H

    def hx(self,X):
        # State space vector
        x = X[0,0] - self.x_rad
        y = X[1,0] - self.y_rad
        z = X[2,0] - self.z_rad
        # Measurements
        r     = sqrt(x**2 + y**2 + z**2)
        theta = atan2(y,x)
        phi   = atan2(z,sqrt(x**2 + y**2))
        # Measurement vector
        Z_k = np.array([[r,theta,phi]]).T
        return Z_k


class RadarFilterCA(RadarModel):

    def __init__(self,dim_x, dim_z,
                      x0  = 1e-6, y0  = 1e-6, z0  = 1e-6,
                      vx0 = 1e-6, vy0 = 1e-6, vz0 = 1e-6,
                      ax0 = 1e-6, ay0 = 1e-6, az0 = 1e-6,
                      dt = 1., std_r = 1., std_theta = 0.05, std_phi = 0.05,
                      x_rad = 0., y_rad = 0., z_rad = 0.):

        dt2 = dt**2/2
        F = np.array([[1, 0, 0,dt, 0, 0,dt2,  0,  0],
                      [0, 1, 0, 0,dt, 0,  0,dt2,  0],
                      [0, 0, 1, 0, 0,dt,  0,  0,dt2],
                      [0, 0, 0, 1, 0, 0, dt,  0,  0],
                      [0, 0, 0, 0, 1, 0,  0, dt,  0],
                      [0, 0, 0, 0, 0, 1,  0,  0, dt],
                      [0, 0, 0, 0, 0, 0,  1,  0,  0],
                      [0, 0, 0, 0, 0, 0,  0,  1,  0],
                      [0, 0, 0, 0, 0, 0,  0,  0,  1]])

        RadarModel.__init__(self, dim_x = dim_x, dim_z = dim_z, F = F,
                            x0  = x0,  y0  = y0,  z0  = z0,
                            vx0 = vx0, vy0 = vy0, vz0 = vz0,
                            ax0 = ax0, ay0 = ay0, az0 = az0,
                            dt = dt, std_r = std_r, std_theta = std_theta, std_phi = std_phi,
                            x_rad = x_rad, y_rad = y_rad, z_rad = z_rad)

        self.x = np.array([[x0,y0,z0,vx0,vy0,vz0,ax0,ay0,az0]]).T

        self.Q = np.zeros((9,9))
        self.Q[0:3,0:3] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.01)
        self.Q[3:6,3:6] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.01)
        self.Q[6:9,6:9] = Q_discrete_white_noise(dim = 3, dt=dt, var= 0.01)


    def HJacob(self,X):
        x = X[0,0] - self.x_rad
        y = X[1,0] - self.y_rad
        z = X[2,0] - self.z_rad
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), y/sqrt(x**2 + y**2 + z**2), z/sqrt(x**2 + y**2 + z**2), 0, 0, 0, 0, 0, 0],
                      [-y/(x**2 + y**2), x/(x**2 + y**2), 0, 0, 0, 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0, 0, 0, 0, 0]])
        return H

    def hx(self,X):
        # State space vector
        x = X[0,0] - self.x_rad
        y = X[1,0] - self.y_rad
        z = X[2,0] - self.z_rad
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
    hjac = hx.jacobian(Matrix([x, y, z, vx, vy, vz]))
    hjac = hx.jacobian(Matrix([x, y, z, vx, vy, vz, ax, ay, az]))
    # For F, the matrices are not changing over time so no need for jacobians

    # ========================= [ARIF2017] model ===============================
    # For f (h is the same as above)
    fxu = Matrix([[x / sympy.sqrt(x**2 + y**2 + z**2) - y - x * z / sympy.sqrt(x**2 + y**2)],
                  [y / sympy.sqrt(x**2 + y**2 + z**2) - y - y * z / sympy.sqrt(x**2 + y**2)],
                  [z / sympy.sqrt(x**2 + y**2 + z**2) + sympy.sqrt(x**2 + y**2)],
                  [-2*y / sympy.sqrt(x**2 + y**2 + z**2) - 2*x*z / sympy.sqrt(x**2 + y**2 + z**2) / sympy.sqrt(x**2 + y**2) -2*x + 2*y*z / sympy.sqrt(x**2 + y**2)],
                  [-2*x / sympy.sqrt(x**2 + y**2 + z**2) - 2*y*z / sympy.sqrt(x**2 + y**2 + z**2) / sympy.sqrt(x**2 + y**2) -2*y + 2*x*z / sympy.sqrt(x**2 + y**2)],
                  [2 * sympy.sqrt(x**2 + y**2) / sympy.sqrt(x**2 + y**2 + z**2)]])
    fjac = fxu.jacobian(Matrix([x, y, z, vx, vy, vz]))
    # ==========================================================================
