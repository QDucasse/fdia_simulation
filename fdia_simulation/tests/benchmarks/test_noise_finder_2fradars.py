# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:04:12 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC, abstractmethod
from filterpy.kalman                  import IMMEstimator
from fdia_simulation.models           import FrequencyRadar
from fdia_simulation.filters          import MultipleFreqRadarsFilterCA,MultipleFreqRadarsFilterCV,MultipleFreqRadarsFilterCT,MultipleFreqRadarsFilterTA
from fdia_simulation.benchmarks       import Benchmark, NoiseFinder2Radars
from fdia_simulation.tests.benchmarks import NoiseFinder2RadarsTestEnv


class NoiseFinder2FreqRadarsTestEnv(NoiseFinder2RadarsTestEnv):

    def setUp_radars_states(self):
        # Radars definitions
        dt_rad1 = 0.6
        self.radar1 = FrequencyRadar(x=2000,y=2000,dt=dt_rad1)
        self.radar1.step = 2
        dt_rad2 = 0.3
        self.radar2 = FrequencyRadar(x=1000,y=1000,dt=dt_rad2)
        self.radar2.step = 1
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_initialization_noise_finder(self):
        self.assertEqual(self.process_noise_finder.radars, [FrequencyRadar(x=2000,y=2000,dt=0.6),FrequencyRadar(x=1000,y=1000,dt=0.3)])
        self.assertTrue(np.array_equal(self.process_noise_finder.states,np.array([[i,i/2,i/10]*3 for i in range(100)])))

    def test_compute_nees(self):
        self.assertEqual(150,len(self.process_noise_finder.compute_nees(10)))


class NoiseFinder2FreqRadarsCATestCase(NoiseFinder2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleFreqRadarsFilterCA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder2Radars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2FreqRadarsCVTestCase(NoiseFinder2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleFreqRadarsFilterCV
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder2Radars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2FreqRadarsCTTestCase(NoiseFinder2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleFreqRadarsFilterCT
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder2Radars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2FreqRadarsTATestCase(NoiseFinder2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultipleFreqRadarsFilterTA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder2Radars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]


if __name__ == "__main__":
    unittest.main()
