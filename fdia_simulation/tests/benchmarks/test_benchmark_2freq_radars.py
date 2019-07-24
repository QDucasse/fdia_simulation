# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:58:06 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC, abstractmethod
from filterpy.kalman                  import IMMEstimator
from fdia_simulation.models           import Radar, FrequencyRadar
from fdia_simulation.filters          import MultipleFreqRadarsFilterCA,MultipleFreqRadarsFilterCV,MultipleFreqRadarsFilterCT,MultipleFreqRadarsFilterTA
from fdia_simulation.benchmarks       import Benchmark
from fdia_simulation.tests.benchmarks import Benchmark1RadarTestEnv, Benchmark2RadarsTestEnv

class Benchmark2FreqRadarsTestEnv(Benchmark2RadarsTestEnv):

    @abstractmethod
    def setUp(self):
        pass

    def setUp_radar_states(self):
        # Radar definition
        dt_rad1 = 0.5
        self.radar1 = FrequencyRadar(x=2000,y=2000,dt=dt_rad1)
        dt_rad2 = 0.4
        self.radar2 = FrequencyRadar(x=1000,y=1000,dt=dt_rad2)
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_gen_data_set(self):
        self.benchmark.gen_data_set()
        self.assertEqual(self.benchmark.measured_values.size, 0)
        self.assertEqual(len(self.benchmark.labeled_values),200)

    def test_process_filter_correct_estimated_positions(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.estimated_positions), (200,3))

    def test_process_filter_nees_true(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.nees), (200,1))

class Benchmark2FreqRadarsCATestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleFreqRadarsFilterCA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2FreqRadarsCVTestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleFreqRadarsFilterCV(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2FreqRadarsCTTestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleFreqRadarsFilterCT(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2FreqRadarsTATestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultipleFreqRadarsFilterTA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2FreqRadarsIMM2TestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleFreqRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleFreqRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
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
        self.assertEqual(np.shape(self.benchmark.probs), (200,2))


class Benchmark2FreqRadarsIMM3TestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleFreqRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleFreqRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultipleFreqRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
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
        self.assertEqual(np.shape(self.benchmark.probs), (200,3))

class Benchmark2FreqRadarsIMM4TestCase(Benchmark2FreqRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultipleFreqRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultipleFreqRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultipleFreqRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ta = MultipleFreqRadarsFilterTA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
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
        self.assertEqual(np.shape(self.benchmark.probs), (200,4))
