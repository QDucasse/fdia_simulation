# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 09:19:36 2019

@author: qde
"""

import unittest
import numpy as np
from pprint import pprint
from copy                    import deepcopy
from math                    import sqrt,atan2, exp
from nose.tools              import raises
from numpy.linalg            import inv
from scipy.linalg            import block_diag
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarFilterTA, MultipleRadarsFilterTA

class RadarFilterTATestCase(unittest.TestCase):
    def setUp(self):
        self.radar     = Radar(x=0,y=0)
        self.q         = 10.
        self.filter_ta = RadarFilterTA(dim_x = 9, dim_z = 3, q = self.q,radar = self.radar)


    # ==========================================================================
    # ========================= Initialization tests ===========================

    def test_initial_F(self):
        dt = self.filter_ta.dt
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
        self.assertTrue(np.array_equal(self.filter_ta.F,F))

    def test_initial_Q(self):
        dt = self.filter_ta.dt
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
        self.assertTrue(np.array_equal(self.filter_ta.Q,Q))

    def test_initial_R(self):
        dt = self.filter_ta.dt
        R = np.array([[1., 0.   , 0.   ],
                      [0., 0.001, 0.   ],
                      [0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.filter_ta.R,R))

    def test_initial_positions(self):
        x0 = self.filter_ta.x[0,0]
        y0 = self.filter_ta.x[3,0]
        z0 = self.filter_ta.x[6,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_ta.x[1,0]
        vy0 = self.filter_ta.x[4,0]
        vz0 = self.filter_ta.x[7,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_accelerations(self):
        vx0 = self.filter_ta.x[2,0]
        vy0 = self.filter_ta.x[5,0]
        vz0 = self.filter_ta.x[8,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_radar_positions(self):
        x_rad = self.filter_ta.x_rad
        y_rad = self.filter_ta.y_rad
        z_rad = self.filter_ta.z_rad
        self.assertEqual(x_rad, 0.)
        self.assertEqual(y_rad, 0.)
        self.assertEqual(z_rad, 0.)

    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        self.filter_ta.dt = dt
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

        computed_F = self.filter_ta.compute_F(self.filter_ta.x)
        self.assertTrue(np.array_equal(self.filter_ta.F,F))
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
        self.filter_ta.dt = dt
        computed_Q = self.filter_ta.compute_Q(q)
        self.assertTrue(np.array_equal(self.filter_ta.Q,Q))
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

        computed_H = self.filter_ta.HJacob(X)
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
        computed_Zk = self.filter_ta.hx(X)
        self.assertTrue(np.array_equal(Zk,computed_Zk))

    # ==========================================================================
    # ========================= predict/update cycle tests =====================

    def test_residual_of(self):
        X       = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        X_prior = np.array([[2000, 200, 20, 2000, 200, 20, 8000, 2, 10]]).T
        z       = np.array([[200, 10, 10]]).T
        computed_resid   = z - self.filter_ta.HJacob(X)@X_prior

        self.filter_ta.x       = X
        self.filter_ta.x_prior = X_prior
        resid = self.filter_ta.residual_of(z)

        self.assertTrue(np.array_equal(computed_resid,resid))

    def test_predict(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        filt = self.filter_ta
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
        filt = self.filter_ta
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


class MultipleRadarsTATestCase(unittest.TestCase):
    def setUp(self):
        self.radar1 = Radar(x=800,y=800)
        self.radar2 = Radar(x=200,y=200)
        radars = [self.radar1,self.radar2]
        self.multiple_ta = MultipleRadarsFilterTA(dim_x = 9, dim_z = 3, q = 1., radars = radars,
                                                  x0 = 100, y0 = 100)

    # ==========================================================================
    # ========================= Initialization tests ===========================
    def test_initial_radar_positions(self):
        positions = [[self.radar1.x,self.radar1.y,self.radar1.z],[self.radar2.x,self.radar2.y,self.radar2.z]]
        computed_positions = self.multiple_ta.radar_positions
        self.assertEqual(computed_positions,positions)

    def test_initial_R(self):
        dt = self.multiple_ta.dt
        R = np.array([[1., 0.   , 0.   , 0., 0.   , 0.   ],
                      [0., 0.001, 0.   , 0., 0.   , 0.   ],
                      [0., 0.   , 0.001, 0., 0.   , 0.   ],
                      [0., 0.   , 0.   , 1., 0.   , 0.   ],
                      [0., 0.   , 0.   , 0., 0.001, 0.   ],
                      [0., 0.   , 0.   , 0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.multiple_ta.R,R))

    def test_initial_F(self):
        dt = self.multiple_ta.dt
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
        self.assertTrue(np.array_equal(self.multiple_ta.F,F))


    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        self.multiple_ta.dt = dt
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

        computed_F = self.multiple_ta.compute_F(self.multiple_ta.x)
        self.assertTrue(np.array_equal(self.multiple_ta.F,F))
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
        self.multiple_ta.dt = dt
        computed_Q = self.multiple_ta.compute_Q(q)
        self.assertTrue(np.array_equal(self.multiple_ta.Q,Q))
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

        computed_H = self.multiple_ta.HJacob(X)
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
        computed_Zk = self.multiple_ta.hx(X)
        self.assertTrue(np.array_equal(Zk,computed_Zk))

    # ==========================================================================
    # ========================= predict/update cycle tests =====================

    def test_residual_of(self):
        X       = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        X_prior = np.array([[2000, 200, 20, 2000, 200, 20, 8000, 2, 10]]).T
        z       = np.array([[200, 10, 10, 210, 9, 8]]).T
        computed_resid   = z - self.multiple_ta.HJacob(X)@X_prior

        self.multiple_ta.x       = X
        self.multiple_ta.x_prior = X_prior
        resid = self.multiple_ta.residual_of(z)

        self.assertTrue(np.array_equal(computed_resid,resid))

    def test_predict(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        filt = self.multiple_ta
        filt.x = X
        predicted_X = filt.F@filt.x
        predicted_P = filt.F@filt.P@filt.F.T + filt.Q

        filt.predict()
        self.assertTrue(np.array_equal(predicted_X,filt.x))
        self.assertTrue(np.array_equal(predicted_P,filt.P))
        self.assertTrue(np.array_equal(predicted_X,filt.x_prior))
        self.assertTrue(np.array_equal(predicted_P,filt.P_prior))

    def test_update(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        z = np.array([[200, 10, 10, 210, 9, 8]]).T
        filt = self.multiple_ta
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
        self.assertTrue(np.allclose(filt.P,new_P))
        self.assertTrue(np.allclose(filt.x,new_X))

if __name__ == "__main__":
    unittest.main()
