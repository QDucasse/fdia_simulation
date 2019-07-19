# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:24:37 2019

@author: qde
"""

import unittest
import numpy as np
from pprint import pprint
from math                    import sqrt,atan2
from nose.tools              import raises
from numpy.linalg            import inv
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarModel, RadarFilterCA, MultipleRadarsFilterCA


class RadarFilterCATestCase(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(x=0,y=0)
        self.filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 1.,radar = self.radar)


    # ==========================================================================
    # ========================= Initialization tests ===========================

    def test_initial_F(self):
        dt = self.filter_ca.dt
        dt2 = dt**2/2
        F = np.array([[1, dt,dt2,  0,  0,  0,  0,  0,  0],
                      [0,  1, dt,  0,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  1, dt,dt2,  0,  0,  0],
                      [0,  0,  0,  0,  1, dt,  0,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  1, dt,dt2],
                      [0,  0,  0,  0,  0,  0,  0,  1, dt],
                      [0,  0,  0,  0,  0,  0,  0,  0,  1]])
        self.assertTrue(np.array_equal(self.filter_ca.F,F))

    def test_initial_R(self):
        dt = self.filter_ca.dt
        R = np.array([[1., 0.   , 0.   ],
                      [0., 0.001, 0.   ],
                      [0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.filter_ca.R,R))

    def test_initial_positions(self):
        x0 = self.filter_ca.x[0,0]
        y0 = self.filter_ca.x[3,0]
        z0 = self.filter_ca.x[6,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_ca.x[1,0]
        vy0 = self.filter_ca.x[4,0]
        vz0 = self.filter_ca.x[7,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_accelerations(self):
        vx0 = self.filter_ca.x[2,0]
        vy0 = self.filter_ca.x[5,0]
        vz0 = self.filter_ca.x[8,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_radar_positions(self):
        x_rad = self.filter_ca.x_rad
        y_rad = self.filter_ca.y_rad
        z_rad = self.filter_ca.z_rad
        self.assertEqual(x_rad, 0.)
        self.assertEqual(y_rad, 0.)
        self.assertEqual(z_rad, 0.)

    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        dt2 = dt**2/2
        F = np.array([[1, dt,dt2,  0,  0,  0,  0,  0,  0],
                      [0,  1, dt,  0,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  1, dt,dt2,  0,  0,  0],
                      [0,  0,  0,  0,  1, dt,  0,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  1, dt,dt2],
                      [0,  0,  0,  0,  0,  0,  0,  1, dt],
                      [0,  0,  0,  0,  0,  0,  0,  0,  1]])
        self.filter_ca.dt = dt
        computed_F = self.filter_ca.compute_F(self.filter_ca.x)
        self.assertTrue(np.array_equal(self.filter_ca.F,F))
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
        self.filter_ca.dt = dt
        computed_Q = self.filter_ca.compute_Q(q)
        self.assertTrue(np.array_equal(self.filter_ca.Q,Q))
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

        computed_H = self.filter_ca.HJacob(X)
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
        computed_Zk = self.filter_ca.hx(X)
        self.assertTrue(np.array_equal(Zk,computed_Zk))

    # ==========================================================================
    # ========================= predict/update cycle tests =====================

    def test_residual_of(self):
        X       = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        X_prior = np.array([[2000, 200, 20, 2000, 200, 20, 8000, 2, 10]]).T
        z       = np.array([[200, 10, 10]]).T
        computed_resid   = z - self.filter_ca.HJacob(X)@X_prior

        self.filter_ca.x       = X
        self.filter_ca.x_prior = X_prior
        resid = self.filter_ca.residual_of(z)

        self.assertTrue(np.array_equal(computed_resid,resid))

    def test_predict(self):
        X = np.array([[1000, 100, 10, 1000, 100, 10, 8000, 2, 10]]).T
        filt = self.filter_ca
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
        z = np.array([[200, 10, 10]]).T
        filt = self.filter_ca
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


class MultipleRadarsCATestCase(unittest.TestCase):
    def setUp(self):
        self.radar1 = Radar(x=800,y=800)
        self.radar2 = Radar(x=200,y=200)
        radars = [self.radar1,self.radar2]
        self.multiple_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 1., radars = radars,
                                                  x0 = 100, y0 = 100)

    # ==========================================================================
    # ========================= Initialization tests ===========================

    def test_initial_F(self):
        dt = self.multiple_ca.dt
        dt2 = dt**2/2
        F = np.array([[1, dt,dt2,  0,  0,  0,  0,  0,  0],
                      [0,  1, dt,  0,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  1, dt,dt2,  0,  0,  0],
                      [0,  0,  0,  0,  1, dt,  0,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  1, dt,dt2],
                      [0,  0,  0,  0,  0,  0,  0,  1, dt],
                      [0,  0,  0,  0,  0,  0,  0,  0,  1]])
        self.assertTrue(np.array_equal(self.multiple_ca.F,F))

    # ==========================================================================
    # ========================= Q/F generation tests ===========================

    def test_F_computing(self):
        dt = 5.
        dt2 = dt**2/2
        F = np.array([[1, dt,dt2,  0,  0,  0,  0,  0,  0],
                      [0,  1, dt,  0,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  1, dt,dt2,  0,  0,  0],
                      [0,  0,  0,  0,  1, dt,  0,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  1, dt,dt2],
                      [0,  0,  0,  0,  0,  0,  0,  1, dt],
                      [0,  0,  0,  0,  0,  0,  0,  0,  1]])
        self.multiple_ca.dt = dt
        computed_F = self.multiple_ca.compute_F(self.multiple_ca.x)
        self.assertTrue(np.array_equal(self.multiple_ca.F,F))
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
        self.multiple_ca.dt = dt
        computed_Q = self.multiple_ca.compute_Q(q)
        self.assertTrue(np.array_equal(self.multiple_ca.Q,Q))
        self.assertTrue(np.array_equal(computed_Q,Q))
