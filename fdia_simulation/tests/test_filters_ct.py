# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:19:47 2019

@author: qde
"""

import unittest
import numpy as np
from pprint import pprint
from copy                    import deepcopy
from math                    import sqrt,atan2, cos, sin
from nose.tools              import raises
from numpy.linalg            import inv
from scipy.linalg            import block_diag
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarFilterCT, MultipleRadarsFilterCT


class RadarFilterCTTestCase(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(x=0,y=0)
        self.q = 10.
        self.filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = self.q,
                                       radar = self.radar)


    # ==========================================================================
    # ========================= Initialization tests ===========================

    def test_initial_F(self):
        dt = self.filter_ct.dt
        X  = self.filter_ct.x
        vx = X[1,0]
        ax = X[2,0]
        vy = X[4,0]
        ay = X[5,0]
        vz = X[7,0]
        az = X[8,0]
        omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
        F_block = np.array([[1,  sin(omega*dt)/omega, (1 - cos(omega*dt))/omega**2],
                            [0,        cos(omega*dt),          sin(omega*dt)/omega],
                            [0, -omega*sin(omega*dt),                cos(omega*dt)]])
        F = block_diag(F_block,F_block,F_block)
        self.assertTrue(np.array_equal(self.filter_ct.F,F))

    def test_initial_R(self):
        dt = self.filter_ct.dt
        R = np.array([[1., 0.   , 0.   ],
                      [0., 0.001, 0.   ],
                      [0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.filter_ct.R,R))

    def test_initial_Q(self):
        dt = self.filter_ct.dt
        q = self.q
        Q = q*np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0,dt, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,dt, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,dt]])
        self.assertTrue(np.array_equal(self.filter_ct.Q,Q))

    def test_initial_positions(self):
        x0 = self.filter_ct.x[0,0]
        y0 = self.filter_ct.x[3,0]
        z0 = self.filter_ct.x[6,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_ct.x[1,0]
        vy0 = self.filter_ct.x[4,0]
        vz0 = self.filter_ct.x[7,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_accelerations(self):
        vx0 = self.filter_ct.x[2,0]
        vy0 = self.filter_ct.x[5,0]
        vz0 = self.filter_ct.x[8,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_radar_positions(self):
        x_rad = self.filter_ct.x_rad
        y_rad = self.filter_ct.y_rad
        z_rad = self.filter_ct.z_rad
        self.assertEqual(x_rad, 0.)
        self.assertEqual(y_rad, 0.)
        self.assertEqual(z_rad, 0.)

    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        X  = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        self.filter_ct.dt = dt
        self.filter_ct.x  = X
        vx = X[1,0]
        ax = X[2,0]
        vy = X[4,0]
        ay = X[5,0]
        vz = X[7,0]
        az = X[8,0]
        omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
        F_block = np.array([[1,  sin(omega*dt)/omega, (1 - cos(omega*dt))/omega**2],
                            [0,        cos(omega*dt),          sin(omega*dt)/omega],
                            [0, -omega*sin(omega*dt),                cos(omega*dt)]])
        F = block_diag(F_block,F_block,F_block)

        computed_F = self.filter_ct.compute_F(self.filter_ct.x)
        self.assertTrue(np.array_equal(self.filter_ct.F,F))
        self.assertTrue(np.array_equal(computed_F,F))

    def test_Q_computing(self):
        dt = 5.
        q  = 20.
        Q = q*np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0,dt, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,dt, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,dt]])
        self.filter_ct.dt = dt
        computed_Q = self.filter_ct.compute_Q(q)
        self.assertTrue(np.array_equal(self.filter_ct.Q,Q))
        self.assertTrue(np.array_equal(computed_Q,Q))

    # ==========================================================================
    # ========================= hx/HJacob tests ================================
    def test_HJacob_computing(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        x = X[0,0]
        y = X[3,0]
        z = X[6,0]
        H = np.array([[x/sqrt(x**2 + y**2 + z**2), 0, 0, y/sqrt(x**2 + y**2 + z**2), 0, 0, z/sqrt(x**2 + y**2 + z**2),0 ,0],
                      [-y/(x**2 + y**2), 0, 0, x/(x**2 + y**2), 0, 0, 0, 0, 0],
                      [-x*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, -y*z/(sqrt(x**2 + y**2)*(x**2 + y**2 + z**2)), 0, 0, sqrt(x**2 + y**2)/(x**2 + y**2 + z**2), 0, 0]])

        computed_H = self.filter_ct.HJacob(X)
        self.assertTrue(np.array_equal(computed_H,H))

    def test_hx_computing(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        x = X[0,0]
        y = X[3,0]
        z = X[6,0]
        r     = sqrt(x**2 + y**2 + z**2)
        theta = atan2(y,x)
        phi   = atan2(z,sqrt(x**2 + y**2))
        Zk    = np.array([[r,theta,phi]]).T
        computed_Zk = self.filter_ct.hx(X)
        self.assertTrue(np.array_equal(Zk,computed_Zk))

    # ==========================================================================
    # ========================= predict/update cycle tests =====================

    def test_residual_of(self):
        X       = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        X_prior = np.array([[2000, 200, 20, 2000, 200, 20, 8000, 2, 10]]).T
        z       = np.array([[200, 10, 10]]).T
        computed_resid   = z - self.filter_ct.HJacob(X)@X_prior

        self.filter_ct.x       = X
        self.filter_ct.x_prior = X_prior
        resid = self.filter_ct.residual_of(z)

        self.assertTrue(np.array_equal(computed_resid,resid))

    def test_predict(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        filt = self.filter_ct
        filt.x = X
        pre_F = deepcopy(filt.F)
        F = filt.compute_F(X)
        P = filt.P
        Q = filt.Q
        predicted_X = F@X
        predicted_P = F@P@F.T + Q

        filt.F = pre_F # Needed to keep F unaltered as before the predict step
        filt.predict()
        self.assertTrue(np.array_equal(predicted_X,filt.x))
        self.assertTrue(np.array_equal(predicted_P,filt.P))
        self.assertTrue(np.array_equal(predicted_X,filt.x_prior))
        self.assertTrue(np.array_equal(predicted_P,filt.P_prior))


    def test_update(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        z = np.array([[200, 10, 10]]).T
        filt = self.filter_ct
        filt.x = X
        filt.predict()
        H = filt.HJacob(filt.x)
        S = H@filt.P@H.T + filt.R
        K = filt.P@H.T@inv(S)

        hx = filt.hx(filt.x)
        y = z - hx
        new_X = filt.x + K@y
        IKH = (filt._I - K@H)
        new_P = (IKH@filt.P)@IKH.T + (K@filt.R)@K.T

        filt.update(z)
        self.assertTrue(np.array_equal(filt.P,new_P))
        self.assertTrue(np.array_equal(filt.x,new_X))


class MultipleRadarsCTTestCase(unittest.TestCase):
    def setUp(self):
        self.radar1 = Radar(x=800,y=800)
        self.radar2 = Radar(x=200,y=200)
        radars = [self.radar1,self.radar2]
        self.q = 10.
        self.multiple_ct = MultipleRadarsFilterCT(dim_x = 9, dim_z = 3, q = self.q, radars = radars,
                                                  x0 = 100, y0 = 100)

    # ==========================================================================
    # ========================= Initialization tests ===========================
    def test_initial_radar_positions(self):
        positions = [[self.radar1.x,self.radar1.y,self.radar1.z],[self.radar2.x,self.radar2.y,self.radar2.z]]
        computed_positions = self.multiple_ct.radar_positions
        self.assertEqual(computed_positions,positions)

    def test_initial_R(self):
        dt = self.multiple_ct.dt
        R = np.array([[1., 0.   , 0.   , 0., 0.   , 0.   ],
                      [0., 0.001, 0.   , 0., 0.   , 0.   ],
                      [0., 0.   , 0.001, 0., 0.   , 0.   ],
                      [0., 0.   , 0.   , 1., 0.   , 0.   ],
                      [0., 0.   , 0.   , 0., 0.001, 0.   ],
                      [0., 0.   , 0.   , 0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.multiple_ct.R,R))

    def test_initial_F(self):
        dt = self.multiple_ct.dt
        X  = self.multiple_ct.x
        vx = X[1,0]
        ax = X[2,0]
        vy = X[4,0]
        ay = X[5,0]
        vz = X[7,0]
        az = X[8,0]
        omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
        F_block = np.array([[1,  sin(omega*dt)/omega, (1 - cos(omega*dt))/omega**2],
                            [0,        cos(omega*dt),          sin(omega*dt)/omega],
                            [0, -omega*sin(omega*dt),                cos(omega*dt)]])
        F = block_diag(F_block,F_block,F_block)
        self.assertTrue(np.array_equal(self.multiple_ct.F,F))

    def test_initial_Q(self):
        dt = self.multiple_ct.dt
        q = self.q
        Q = q*np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0,dt, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,dt, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,dt]])
        self.assertTrue(np.array_equal(self.multiple_ct.Q,Q))


    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        X  = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        self.multiple_ct.x  = X
        self.multiple_ct.dt = dt
        vx = X[1,0]
        ax = X[2,0]
        vy = X[4,0]
        ay = X[5,0]
        vz = X[7,0]
        az = X[8,0]
        omega = sqrt(ax**2 + ay**2 + az**2)/sqrt(vx**2 + vy**2 + vz**2)
        F_block = np.array([[1,  sin(omega*dt)/omega, (1 - cos(omega*dt))/omega**2],
                            [0,        cos(omega*dt),          sin(omega*dt)/omega],
                            [0, -omega*sin(omega*dt),                cos(omega*dt)]])
        F = block_diag(F_block,F_block,F_block)

        computed_F = self.multiple_ct.compute_F(self.multiple_ct.x)
        self.assertTrue(np.array_equal(self.multiple_ct.F,F))
        self.assertTrue(np.array_equal(computed_F,F))

    def test_Q_computing(self):
        dt = 5.
        q  = 20.
        Q = q*np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0,dt, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0,dt, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0,dt]])
        self.multiple_ct.dt = dt
        computed_Q = self.multiple_ct.compute_Q(q)
        self.assertTrue(np.array_equal(self.multiple_ct.Q,Q))
        self.assertTrue(np.array_equal(computed_Q,Q))

    # ==========================================================================
    # ============================= HJacob/hx generation =======================

    def test_HJacob_computing(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        x1 = X[0,0] - self.radar1.x
        y1 = X[3,0] - self.radar1.y
        z1 = X[6,0] - self.radar1.z
        x2 = X[0,0] - self.radar2.x
        y2 = X[3,0] - self.radar2.y
        z2 = X[6,0] - self.radar2.z
        H = np.array([[x1/sqrt(x1**2 + y1**2 + z1**2), 0, 0, y1/sqrt(x1**2 + y1**2 + z1**2), 0, 0, z1/sqrt(x1**2 + y1**2 + z1**2),0 ,0],
                      [-y1/(x1**2 + y1**2), 0, 0, x1/(x1**2 + y1**2), 0, 0, 0, 0, 0],
                      [-x1*z1/(sqrt(x1**2 + y1**2)*(x1**2 + y1**2 + z1**2)), 0, 0, -y1*z1/(sqrt(x1**2 + y1**2)*(x1**2 + y1**2 + z1**2)), 0, 0, sqrt(x1**2 + y1**2)/(x1**2 + y1**2 + z1**2), 0, 0],
                      [x2/sqrt(x2**2 + y2**2 + z2**2), 0, 0, y2/sqrt(x2**2 + y2**2 + z2**2), 0, 0, z2/sqrt(x2**2 + y2**2 + z2**2),0 ,0],
                      [-y2/(x2**2 + y2**2), 0, 0, x2/(x2**2 + y2**2), 0, 0, 0, 0, 0],
                      [-x2*z2/(sqrt(x2**2 + y2**2)*(x2**2 + y2**2 + z2**2)), 0, 0, -y2*z2/(sqrt(x2**2 + y2**2)*(x2**2 + y2**2 + z2**2)), 0, 0, sqrt(x2**2 + y2**2)/(x2**2 + y2**2 + z2**2), 0, 0]])

        computed_H = self.multiple_ct.HJacob(X)
        self.assertTrue(np.array_equal(computed_H,H))

    def test_hx_computing(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        x1 = X[0,0] - self.radar1.x
        y1 = X[3,0] - self.radar1.y
        z1 = X[6,0] - self.radar1.z
        x2 = X[0,0] - self.radar2.x
        y2 = X[3,0] - self.radar2.y
        z2 = X[6,0] - self.radar2.z
        r1     = sqrt(x1**2 + y1**2 + z1**2)
        theta1 = atan2(y1,x1)
        phi1   = atan2(z1,sqrt(x1**2 + y1**2))
        r2     = sqrt(x2**2 + y2**2 + z2**2)
        theta2 = atan2(y2,x2)
        phi2   = atan2(z2,sqrt(x2**2 + y2**2))
        Zk     = np.array([[r1,theta1,phi1,r2,theta2,phi2]]).T
        computed_Zk = self.multiple_ct.hx(X)
        self.assertTrue(np.array_equal(Zk,computed_Zk))

    # ==========================================================================
    # ========================= predict/update cycle tests =====================

    def test_residual_of(self):
        X       = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        X_prior = np.array([[2000, 200, 20, 2000, 200, 20, 8000, 2, 10]]).T
        z       = np.array([[200, 10, 10, 210, 9, 8]]).T
        computed_resid   = z - self.multiple_ct.HJacob(X)@X_prior

        self.multiple_ct.x       = X
        self.multiple_ct.x_prior = X_prior
        resid = self.multiple_ct.residual_of(z)

        self.assertTrue(np.array_equal(computed_resid,resid))

    def test_predict(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        filt = self.multiple_ct
        filt.x = X
        pre_F = deepcopy(filt.F)
        F = filt.compute_F(X)
        P = filt.P
        Q = filt.Q
        predicted_X = F@X
        predicted_P = F@P@F.T + Q

        filt.F = pre_F # Needed to keep F unaltered as before the predict step
        filt.predict()
        self.assertTrue(np.array_equal(predicted_X,filt.x))
        self.assertTrue(np.array_equal(predicted_P,filt.P))
        self.assertTrue(np.array_equal(predicted_X,filt.x_prior))
        self.assertTrue(np.array_equal(predicted_P,filt.P_prior))


    def test_update(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        z = np.array([[200, 10, 10, 210, 9, 8]]).T
        filt = self.multiple_ct
        filt.x = X
        filt.predict()
        H = filt.HJacob(filt.x)
        S = H@filt.P@H.T + filt.R
        K = filt.P@H.T@inv(S)

        hx = filt.hx(filt.x)
        y = z - hx
        new_X = filt.x + K@y
        IKH = (filt._I - K@H)
        new_P = (IKH@filt.P)@IKH.T + (K@filt.R)@K.T

        filt.update(z)
        print(filt.P-new_P)
        self.assertTrue(np.allclose(filt.P,new_P))
        self.assertTrue(np.allclose(filt.x,new_X))



if __name__ == "__main__":
    unittest.main()
