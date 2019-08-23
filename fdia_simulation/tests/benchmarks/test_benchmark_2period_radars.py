# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:58:06 2019

@author: qde
"""

import unittest
import numpy as np
from abc                              import ABC
from fdia_simulation.models           import Radar, PeriodRadar
from fdia_simulation.filters          import (MultiplePeriodRadarsFilterCA,
                                              MultiplePeriodRadarsFilterCV,
                                              MultiplePeriodRadarsFilterCT,
                                              MultiplePeriodRadarsFilterTA,RadarIMM)
from fdia_simulation.benchmarks       import Benchmark
from fdia_simulation.tests.benchmarks import Benchmark1RadarTestEnv, Benchmark2RadarsTestEnv

class Benchmark2PeriodRadarsTestEnv(Benchmark2RadarsTestEnv):

    def setUp_radar_states(self):
        # Radar definition
        dt_rad1 = 0.6
        self.radar1 = PeriodRadar(x=2000,y=2000,dt=dt_rad1)
        dt_rad2 = 0.3
        self.radar2 = PeriodRadar(x=1000,y=1000,dt=dt_rad2)
        self.radars = [self.radar1, self.radar2]
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(225)])
        self.len_elements = int(len(self.states)/self.radar1.step + len(self.states)/self.radar2.step)

    def test_gen_data_set(self):
        self.benchmark.gen_data_set()
        self.assertEqual(self.benchmark.measured_values.size, 0)
        self.assertEqual(len(self.benchmark.labeled_values),self.len_elements)

    def test_process_filter_correct_estimated_positions(self):
        self.benchmark.gen_data_set()
        #import pdb; pdb.set_trace()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.estimated_positions), (self.len_elements,3))

    def test_process_filter_nees_true(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.nees), (self.len_elements,1))

class Benchmark2PeriodRadarsCATestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultiplePeriodRadarsFilterCA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2PeriodRadarsCVTestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultiplePeriodRadarsFilterCV(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2PeriodRadarsCTTestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultiplePeriodRadarsFilterCT(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2PeriodRadarsTATestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = MultiplePeriodRadarsFilterTA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
        # Benchmark definitions
        self.benchmark   = Benchmark(radars = self.radars, radar_filter = self.radar_filter,   states = self.states)

class Benchmark2PeriodRadarsIMM2TestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultiplePeriodRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultiplePeriodRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca]
        mu = [0.5, 0.5]
        trans = np.array([[0.999, 0.001],
                          [0.001, 0.999]])
        self.radar_filter = RadarIMM(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (self.len_elements,2))


class Benchmark2PeriodRadarsIMM3TestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultiplePeriodRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultiplePeriodRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultiplePeriodRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct]
        mu = [0.33, 0.33, 0.33]
        trans = np.array([[0.998, 0.001, 0.001],
                          [0.050, 0.900, 0.050],
                          [0.001, 0.001, 0.998]])
        self.radar_filter = RadarIMM(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (self.len_elements,3))

class Benchmark2PeriodRadarsIMM4TestCase(Benchmark2PeriodRadarsTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = MultiplePeriodRadarsFilterCA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_cv = MultiplePeriodRadarsFilterCV(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ct = MultiplePeriodRadarsFilterCT(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        self.radar_filter_ta = MultiplePeriodRadarsFilterTA(dim_x = 9, dim_z = 3, q = 100., radars = self.radars)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct, self.radar_filter_ta]
        mu = [0.25, 0.25, 0.25, 0.25]
        trans = np.array([[0.997, 0.001, 0.001, 0.001],
                          [0.050, 0.850, 0.050, 0.050],
                          [0.001, 0.001, 0.997, 0.001],
                          [0.001, 0.001, 0.001, 0.997]])
        self.radar_filter = RadarIMM(filters = filters, mu = mu, M = trans)

        # Benchmark definitions
        self.benchmark = Benchmark(radars = self.radars, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (self.len_elements,4))


if __name__ == "__main__":
    unittest.main()
