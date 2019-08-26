# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:04:12 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC, abstractmethod
from filterpy.kalman                  import IMMEstimator
from fdia_simulation.models           import PeriodRadar
from fdia_simulation.filters          import MultiplePeriodRadarsFilterCA,MultiplePeriodRadarsFilterCV,MultiplePeriodRadarsFilterCT,MultiplePeriodRadarsFilterTA
from fdia_simulation.benchmarks       import Benchmark, NoiseFinderMultipleRadars
from fdia_simulation.tests.benchmarks import NoiseFinderMultipleRadarsTestEnv


class NoiseFinder2PeriodRadarsTestEnv(NoiseFinderMultipleRadarsTestEnv):

    def setUp_radars_states(self):
        # Radars definitions
        dt_rad1 = 0.01
        self.radar1 = PeriodRadar(x=2000,y=2000,dt=dt_rad1)
        dt_rad2 = 0.02
        self.radar2 = PeriodRadar(x=1000,y=1000,dt=dt_rad2)
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(300)])

    def test_initialization_noise_finder(self):
        self.assertEqual(self.process_noise_finder.radars, [PeriodRadar(x=2000,y=2000,dt=0.01),PeriodRadar(x=1000,y=1000,dt=0.02)])
        self.assertTrue(np.array_equal(self.process_noise_finder.states,np.array([[i,i/2,i/10]*3 for i in range(300)])))

    def test_compute_nees(self):
        self.assertEqual(450,len(self.process_noise_finder.compute_nees(10)))


class NoiseFinder2PeriodRadarsCATestCase(NoiseFinder2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultiplePeriodRadarsFilterCA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2PeriodRadarsCVTestCase(NoiseFinder2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultiplePeriodRadarsFilterCV
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2PeriodRadarsCTTestCase(NoiseFinder2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultiplePeriodRadarsFilterCT
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder2PeriodRadarsTATestCase(NoiseFinder2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = MultiplePeriodRadarsFilterTA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinderMultipleRadars(radars = self.radars,
                                                       states = self.states,
                                                       filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]


if __name__ == "__main__":
    unittest.main()
