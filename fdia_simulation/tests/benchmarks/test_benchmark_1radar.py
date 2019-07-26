# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:58:06 2019

@author: qde
"""

import unittest
import numpy as np
from abc                        import ABC, abstractmethod
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar
from fdia_simulation.filters    import RadarFilterCA,RadarFilterCV,RadarFilterCT,RadarFilterTA
from fdia_simulation.benchmarks import Benchmark


class Benchmark1RadarTestEnv(ABC):

    @abstractmethod
    def setUp(self):
        pass

    def setUp_radar_states(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])

    def test_initialization_radars_1radar(self):
        self.assertEqual([self.radar],self.benchmark.radars)

    def test_initialization_is_imm(self):
        self.assertFalse(self.benchmark.filter_is_imm)

    def test_initialization_measurement_sets(self):
        self.assertFalse(self.benchmark.measured_values)
        self.assertFalse(self.benchmark.labeled_values)

    def test_initialization_pos_data(self):
        pos_data = np.array([[i]*3 for i in range(100)])
        self.assertTrue(np.array_equal(self.benchmark.pos_data,pos_data))

    # ==========================================================================
    # ========================== Function tests ================================

    def test_gen_data_set(self):
        self.benchmark.gen_data_set()
        self.assertFalse(self.benchmark.labeled_values)
        self.assertEqual(np.shape(self.benchmark.measured_values),(100,3))

    def test_process_filter_nees_false(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = False)
        self.assertTrue(self.benchmark.nees.size == 0)

    def test_process_filter_nees_true(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.nees), (100,1))

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertTrue(self.benchmark.probs.size == 0)

    def test_process_filter_correct_estimated_positions(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.estimated_positions), (100,3))

class Benchmark1RadarCATestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CA model
        self.radar_filter = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarCVTestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CV model
        self.radar_filter = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarCTTestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: CT model
        self.radar_filter = RadarFilterCT(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarTATestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition: TA model
        self.radar_filter = RadarFilterTA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)


class Benchmark1RadarIMM2TestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca]
        mu = [0.5, 0.5]
        trans = np.array([[0.999, 0.001],
                          [0.001, 0.999]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,2))


class Benchmark1RadarIMM3TestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct]
        mu = [0.33, 0.33, 0.33]
        trans = np.array([[0.998, 0.001, 0.001],
                          [0.050, 0.900, 0.050],
                          [0.001, 0.001, 0.998]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definition
        self.benchmark = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,3))

class Benchmark1RadarIMM4TestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar & States generation
        self.setUp_radar_states()
        # Filter definition
        ## Classical models
        self.radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_ta = RadarFilterTA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        ## IMM with ca, cv and ct models
        filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct, self.radar_filter_ta]
        mu = [0.25, 0.25, 0.25, 0.25]
        trans = np.array([[0.997, 0.001, 0.001, 0.001],
                          [0.050, 0.850, 0.050, 0.050],
                          [0.001, 0.001, 0.997, 0.001],
                          [0.001, 0.001, 0.001, 0.997]])
        self.radar_filter = IMMEstimator(filters = filters, mu = mu, M = trans)

        # Benchmark definitions
        self.benchmark = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,4))


if __name__ == "__main__":
    unittest.main()
