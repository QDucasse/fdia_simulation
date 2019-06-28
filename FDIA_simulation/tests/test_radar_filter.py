# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:59:32 2019

@author: qde
"""

import unittest
import numpy as np
from nose.tools import raises
from fdia_simulation.filters.radar_filter import RadarModel, RadarFilterCV, RadarFilterCA

class RadarModelTestCase(unittest.TestCase):
    @raises(TypeError)
    def test_no_initialization(self):
        abstractClassInstance = RadarModel()

class RadarFilterCVTestCase(unittest.TestCase):
    def setUp(self):
        self.filter_cv = RadarFilterCV(dim_x = 6, dim_z = 3)

    def test_initial_F(self):
        dt = self.filter_cv.dt
        F = np.array([[1, 0, 0,dt, 0, 0],
                      [0, 1, 0, 0,dt, 0],
                      [0, 0, 1, 0, 0,dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        self.assertTrue(np.array_equal(self.filter_cv.F,F))

    def test_initial_R(self):
        dt = self.filter_cv.dt
        R = np.array([[1., 0.  , 0.  ],
                      [1., 0.01, 0.  ],
                      [1., 0.  , 0.01]])
        self.assertTrue(np.array_equal(self.filter_cv.R,R))

    def test_initial_positions(self):
        x0 = self.filter_cv.x[0,0]
        y0 = self.filter_cv.x[1,0]
        z0 = self.filter_cv.x[2,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_cv.x[3,0]
        vy0 = self.filter_cv.x[4,0]
        vz0 = self.filter_cv.x[5,0]
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
        self.filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3)

    def test_initial_F(self):
        dt = self.filter_ca.dt
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
        self.assertTrue(np.array_equal(self.filter_ca.F,F))

    def test_initial_R(self):
        dt = self.filter_ca.dt
        R = np.array([[1., 0.  , 0.  ],
                      [1., 0.01, 0.  ],
                      [1., 0.  , 0.01]])
        self.assertTrue(np.array_equal(self.filter_ca.R,R))

    def test_initial_positions(self):
        x0 = self.filter_ca.x[0,0]
        y0 = self.filter_ca.x[1,0]
        z0 = self.filter_ca.x[2,0]
        self.assertEqual(x0, 1e-6)
        self.assertEqual(y0, 1e-6)
        self.assertEqual(z0, 1e-6)

    def test_initial_velocities(self):
        vx0 = self.filter_ca.x[3,0]
        vy0 = self.filter_ca.x[4,0]
        vz0 = self.filter_ca.x[5,0]
        self.assertEqual(vx0, 1e-6)
        self.assertEqual(vy0, 1e-6)
        self.assertEqual(vz0, 1e-6)

    def test_initial_accelerations(self):
        ax0 = self.filter_ca.x[6,0]
        ay0 = self.filter_ca.x[7,0]
        az0 = self.filter_ca.x[8,0]
        self.assertEqual(ax0, 1e-6)
        self.assertEqual(ay0, 1e-6)
        self.assertEqual(az0, 1e-6)

    def test_initial_radar_positions(self):
        x_rad = self.filter_ca.x_rad
        y_rad = self.filter_ca.y_rad
        z_rad = self.filter_ca.z_rad
        self.assertEqual(x_rad, 0.)
        self.assertEqual(y_rad, 0.)
        self.assertEqual(z_rad, 0.)
