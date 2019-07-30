# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:40:26 2019

@author: qde
"""


import unittest
import numpy as np
from abc                        import ABC, abstractmethod
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar
from fdia_simulation.filters    import RadarFilterCA,RadarFilterCV,RadarFilterCT,RadarFilterTA
from fdia_simulation.benchmarks import Benchmark, NoiseFinder1Radar


class NoiseFinder1RadarTestEnv(ABC):

    def setUp_radars_states(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_initialization_noise_finder(self):
        self.assertEqual(self.process_noise_finder.radar, Radar(x=2000,y=2000))
        self.assertTrue(np.array_equal(self.process_noise_finder.states,np.array([[i,i/2,i/10]*3 for i in range(100)])))

    def test_compute_nees(self):
        self.assertEqual(100,len(self.process_noise_finder.compute_nees(10)))

    def test_iterate_same_simulation(self):
        self.process_noise_finder.nb_iterations = 3
        self.assertEqual(3,len(self.process_noise_finder.iterate_same_simulation(q = 10)))

    def test_launch_benchmark(self):
        self.process_noise_finder.launch_benchmark()
        self.assertEqual(5,len(self.process_noise_finder.means_nees))

    def test_best_value(self):
        self.process_noise_finder.means_nees = {1.:0.3, 2.:0.4, 2:0.5}
        self.assertEqual(1.,self.process_noise_finder.best_value())

class NoiseFinder1RadarCATestCase(NoiseFinder1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = RadarFilterCA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder1Radar(radar  = self.radar,
                                                      states = self.states,
                                                      filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder1RadarCVTestCase(NoiseFinder1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = RadarFilterCV
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder1Radar(radar  = self.radar,
                                                      states = self.states,
                                                      filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder1RadarCTTestCase(NoiseFinder1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = RadarFilterCT
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder1Radar(radar  = self.radar,
                                                      states = self.states,
                                                      filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]

class NoiseFinder1RadarTATestCase(NoiseFinder1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar and states generation
        self.setUp_radars_states()
        # Filter definition
        self.filter = RadarFilterTA
        # Process noise finder definition
        self.process_noise_finder = NoiseFinder1Radar(radar  = self.radar,
                                                      states = self.states,
                                                      filter = self.filter)
        # Reduction of the actual list for testing purposes
        self.process_noise_finder.TO_TEST = [1.,2.,3.,4.,5.]


if __name__ == "__main__":
    unittest.main()
