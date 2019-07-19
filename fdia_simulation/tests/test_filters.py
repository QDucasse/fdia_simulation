# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:59:32 2019

@author: qde
"""

import unittest
import numpy as np
from nose.tools              import raises
from fdia_simulation.models  import Radar
from fdia_simulation.filters import RadarModel, RadarFilterCA,RadarFilterCV, MultipleRadarsFilterCA,MultipleRadarsFilterCV

class RadarModelTestCase(unittest.TestCase):
    @raises(TypeError)
    def test_no_initialization(self):
        abstractClassInstance = RadarModel()

class RadarFilterCVTestCase(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(x=0,y=0)
        self.filter_cv = RadarFilterCV(dim_x = 6, dim_z = 3, q = 1.,radar = self.radar)

    def test_initial_F(self):
        dt = self.filter_cv.dt
        F = np.array([[1,dt, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1,dt, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1,dt, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(self.filter_cv.F,F))

    def test_initial_R(self):
        dt = self.filter_cv.dt
        R = np.array([[1., 0.   , 0.   ],
                      [0., 0.001, 0.   ],
                      [0., 0.   , 0.001]])
        self.assertTrue(np.array_equal(self.filter_cv.R,R))

    def test_initial_positions(self):
        x0 = self.filter_cv.x[0,0]
        y0 = self.filter_cv.x[3,0]
        z0 = self.filter_cv.x[6,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_cv.x[1,0]
        vy0 = self.filter_cv.x[4,0]
        vz0 = self.filter_cv.x[7,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_accelerations(self):
        vx0 = self.filter_cv.x[2,0]
        vy0 = self.filter_cv.x[5,0]
        vz0 = self.filter_cv.x[8,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_radar_positions(self):
        x_rad = self.filter_cv.x_rad
        y_rad = self.filter_cv.y_rad
        z_rad = self.filter_cv.z_rad
        self.assertEqual(x_rad, 0.)
        self.assertEqual(y_rad, 0.)
        self.assertEqual(z_rad, 0.)




class RadarFilterCATestCase(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(x=0,y=0)
        self.filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 1.,radar = self.radar)

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

class MultipleRadarsCVTestCase(unittest.TestCase):
    def setUp(self):
        self.radar1 = Radar(x=800,y=800)
        self.radar2 = Radar(x=200,y=200)
        radars = [self.radar1,self.radar2]
        self.multiple_cv = MultipleRadarsFilterCV(dim_x = 9, dim_z = 3, q = 1., radars = radars,
                                                  x0 = 100, y0 = 100)

    def test_initial_F(self):
        dt = self.multiple_cv.dt
        dt2 = dt**2/2
        F = np.array([[1, dt,  0,  0,  0,  0,  0,  0,  0],
                      [0,  1,  0,  0,  0,  0,  0,  0,  0],
                      [0,  0,  1,  0,  0,  0,  0,  0,  0],
                      [0,  0,  0,  1, dt,  0,  0,  0,  0],
                      [0,  0,  0,  0,  1,  0,  0,  0,  0],
                      [0,  0,  0,  0,  0,  1,  0,  0,  0],
                      [0,  0,  0,  0,  0,  0,  1, dt,  0],
                      [0,  0,  0,  0,  0,  0,  0,  1,  0],
                      [0,  0,  0,  0,  0,  0,  0,  0,  1]])
        self.assertTrue(np.array_equal(self.multiple_cv.F,F))

class MultipleRadarsCATestCase(unittest.TestCase):
    def setUp(self):
        self.radar1 = Radar(x=800,y=800)
        self.radar2 = Radar(x=200,y=200)
        radars = [self.radar1,self.radar2]
        self.multiple_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 1., radars = radars,
                                                  x0 = 100, y0 = 100)

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

if __name__ == "__main__":
    unittest.main()
