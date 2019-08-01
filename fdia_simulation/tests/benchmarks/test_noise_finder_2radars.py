# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:49:35 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC, abstractmethod
from filterpy.kalman                  import IMMEstimator
from fdia_simulation.models           import Radar
from fdia_simulation.filters          import MultipleRadarsFilterCA,MultipleRadarsFilterCV,MultipleRadarsFilterCT,MultipleRadarsFilterTA
from fdia_simulation.benchmarks       import Benchmark, NoiseFinderMultipleRadars
from fdia_simulation.tests.benchmarks import NoiseFinder1RadarTestEnv


class NoiseFinderMultipleRadarsTestEnv(NoiseFinder1RadarTestEnv):

    def setUp_radars_states(self):
        # Radars definitions
        self.radar1 = Radar(x=2000,y=2000)
        self.radar1.step = 1.
        self.radar2 = Radar(x=1000,y=1000)
        self.radar2.step = 1
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_initialization_noise_finder(self):
        self.assertEqual(self.process_noise_finder.radars, [Radar(x=2000,y=2000),Radar(x=1000,y=1000)])
        self.assertTrue(np.array_equal(self.process_noise_finder.states,np.array([[i,i/2,i/10]*3 for i in range(100)])))


class NoiseFinderMultipleRadarsCATestCase(NoiseFinderMultipleRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleRadarsFilterCA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinderMultipleRadarsCVTestCase(NoiseFinderMultipleRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleRadarsFilterCV
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinderMultipleRadarsCTTestCase(NoiseFinderMultipleRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleRadarsFilterCT
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinderMultipleRadarsTATestCase(NoiseFinderMultipleRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleRadarsFilterTA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]


if __name__ == "__main__":
    unittest.main()
