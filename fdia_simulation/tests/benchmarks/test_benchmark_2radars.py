# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:20:58 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC
from filterpy.kalman                  import IMMEstimator
from fdia_simulation.models           import Radar, FrequencyRadar
from fdia_simulation.filters          import MultipleRadarsFilterCA,MultipleRadarsFilterCV,MultipleRadarsFilterCT,MultipleRadarsFilterTA
from fdia_simulation.benchmarks       import Benchmark
from fdia_simulation.tests.benchmarks import Benchmark1RadarTestEnv

class Benchmark2RadarsTestEnv(Benchmark1RadarTestEnv):

    def setUp_radar_states(self):
        # Radars definitions
        self.radar1 = Radar(x=2000,y=2000)
        self.radar1.step = 1.
        self.radar2 = Radar(x=1000,y=1000)
        self.radar2.step = 1
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_initialization_radars_1radar(self):
        pass

    def test_initialization_radars_2radars(self):
        self.assertEqual([self.radar1, self.radar2],self.benchmark.radars)

    def test_gen_data_set(self):
        self.benchmark.gen_data_set()
        self.assertFalse(self.benchmark.labeled_values)
        self.assertEqual(np.shape(self.benchmark.measured_values),(100,6))

class Benchmark2RadarsCATestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleRadarsFilterCA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2RadarsCVTestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleRadarsFilterCV(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2RadarsCTTestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleRadarsFilterCT(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2RadarsTATestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleRadarsFilterTA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2RadarsIMM2TestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca]
        mu = [0.5, 0.5]
        trans = np.array([[0.999, 0.001],
                          [0.001, 0.999]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,2))


class Benchmark2RadarsIMM3TestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultipleRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct]
        mu = [0.33, 0.33, 0.33]
        trans = np.array([[0.998, 0.001, 0.001],
                          [0.050, 0.900, 0.050],
                          [0.001, 0.001, 0.998]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,3))

class Benchmark2RadarsIMM4TestCase(Benchmark2RadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultipleRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ta = MultipleRadarsFilterTA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct, self.radar_filter_ta]
        mu = [0.25, 0.25, 0.25, 0.25]
        trans = np.array([[0.997, 0.001, 0.001, 0.001],
                          [0.050, 0.850, 0.050, 0.050],
                          [0.001, 0.001, 0.997, 0.001],
                          [0.001, 0.001, 0.001, 0.997]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definitions
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,4))

if __name__ == "__main__":
    unittest.main()
