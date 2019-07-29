# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:20:16 2019

@author: qde
"""

import unittest
import numpy as np
from math                   import sqrt,atan2, isclose
from fdia_simulation.models import Radar, FrequencyRadar

class RadarTestCase(unittest.TestCase):
    def setUp(self):
        self.radar = Radar(x = 200, y = 200,
                           r_std = 1., theta_std = 0.001, phi_std = 0.001)

    # ==========================================================================
    # ========================= Initialization tests ===========================
    def test_initial_r_std(self):
        self.assertEqual(self.radar.r_std,1.)

    def test_initial_theta_std(self):
        self.assertEqual(self.radar.theta_std,0.001)

    def test_initial_phi_std(self):
        self.assertEqual(self.radar.phi_std,0.001)

    def test_initial_position(self):
        self.assertEqual(self.radar.x, 200)
        self.assertEqual(self.radar.y, 200)
        self.assertEqual(self.radar.z, 0)

    def test_initial_step(self):
        dt = 0.1
        DT_TRACK = 0.01
        step = dt/DT_TRACK
        self.assertEqual(self.radar.step, step)

    # ==========================================================================
    # ========================= Initialization tests ===========================

    def test_get_position(self):
        position = [self.radar.x,self.radar.y,self.radar.z]
        self.assertEqual(position, self.radar.get_position())

    def test_sample_position_data(self):
        position_data = np.array([[i,i,i] for i in range(10)])
        self.radar.step = 3
        sample = np.array([[0, 0, 0],[3, 3, 3],[6, 6, 6],[9, 9, 9]])
        computed_sample = self.radar.sample_position_data(position_data)
        self.assertTrue(np.array_equal(sample,computed_sample))

    def test_gen_data_1_position(self):
        position_data = np.array([[100. , 200., 1000.]])
        x = position_data[0][0] - self.radar.x
        y = position_data[0][1] - self.radar.y
        z = position_data[0][2] - self.radar.z
        r     = sqrt(x**2 + y**2 + z**2)
        theta = atan2(y,x)
        phi   = atan2(z, sqrt(x**2 + y**2))
        radar_data = [r], [theta], [phi]
        self.assertEqual(radar_data, self.radar.gen_data(position_data))

    def test_gen_data_2_positions(self):
        position_data = np.array([[100. , 200., 1000.],[110.,210.,1010.]])
        x1 = position_data[0][0] - self.radar.x
        y1 = position_data[0][1] - self.radar.y
        z1 = position_data[0][2] - self.radar.z
        r1     = sqrt(x1**2 + y1**2 + z1**2)
        theta1 = atan2(y1,x1)
        phi1   = atan2(z1, sqrt(x1**2 + y1**2))

        x2 = position_data[1][0] - self.radar.x
        y2 = position_data[1][1] - self.radar.y
        z2 = position_data[1][2] - self.radar.z
        r2     = sqrt(x2**2 + y2**2 + z2**2)
        theta2 = atan2(y2,x2)
        phi2   = atan2(z2, sqrt(x2**2 + y2**2))

        radar_data = [r1, r2], [theta1, theta2], [phi1, phi2]
        self.assertEqual(radar_data, self.radar.gen_data(position_data))

    def test_radar2cartesian(self):
        pass

    def test_radar_pos_no_influence(self):
        position_data = np.array([[i,i,i] for i in range(10)])
        rs,thetas,phis = self.radar.gen_data(position_data)
        xs, ys, zs = self.radar.radar2cartesian(rs,thetas,phis)
        computed_position_data = np.array(list(zip(xs,ys,zs)))
        print(position_data)
        print(computed_position_data)
        self.assertTrue(np.allclose(position_data,computed_position_data))


    # def test_sense(self):
    #     radar_data = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2],[3, 3, 3],[4, 4, 4],
    #                            [5, 5, 5],[6, 6, 6],[7, 7, 7],[8, 8, 8],[9, 9, 9]])
    #     rs     = radar_data[:,0]
    #     thetas = radar_data[:,1]
    #     phis   = radar_data[:,2]
    #     noisy_rs, noisy_thetas, noisy_phis = self.radar.sense(rs,thetas,phis)
    #     print(np.std(noisy_rs))
    #     self.assertTrue(isclose(np.std(noisy_rs),self.radar.r_std))


class FrequencyRadarTestCase(RadarTestCase):
    def setUp(self):
        self.radar = FrequencyRadar(x = 200, y = 200, dt = 0.1,
                                    r_std = 1.,
                                    theta_std = 0.001,
                                    phi_std = 0.001,
                                    time_std = 0.001)

    def test_compute_meas_time(self):
        size = 10
        computed_meas_times = self.radar.compute_meas_times(size)
        self.assertEqual(size, len(computed_meas_times))
        # ex_time = 0
        # for time in computed_meas_times:
        #     self.assertTrue(isclose(time,ex_time,rel_tol = self.radar.time_std))
        #     ex_time = time + self.radar.dt

    def test_compute_measurements_tags(self):
         position_data = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2],[3, 3, 3],[4, 4, 4],
                                   [5, 5, 5],[6, 6, 6],[7, 7, 7],[8, 8, 8],[9, 9, 9]])
         self.radar.tag = 0
         labeled_measurements = self.radar.compute_measurements(position_data)
         for labeled_meas in labeled_measurements:
             self.assertEqual(labeled_meas.tag, self.radar.tag)

if __name__ == "__main__":
    unittest.main()
