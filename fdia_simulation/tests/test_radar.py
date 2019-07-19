# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:20:16 2019

@author: qde
"""

import unittest
from fdia_simulation.models.radar import Radar, ManeuveredAircraft

class RadarTestCase(unittest.TestCase):
    def setUp(self):
        self.radar_test = Radar(r_noise_std = 1., theta_noise_std = 0.1, phi_noise_std = 0.1,x = 50, y = 50)

    def test_initial_r_noise_std(self):
        self.assertEqual(self.radar_test.r_noise_std,1.)

    def test_initial_theta_noise_std(self):
        self.assertEqual(self.radar_test.theta_noise_std,0.1)

    def test_initial_phi_noise_std(self):
        self.assertEqual(self.radar_test.phi_noise_std,0.1)



if __name__ == "__main__":
    unittest.main()
