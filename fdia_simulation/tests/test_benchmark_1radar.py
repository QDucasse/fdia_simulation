# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:58:06 2019

@author: qde
"""

import unittest
import numpy as np
from abc                        import ABC, abstractmethod
from copy                       import deepcopy
from numpy.linalg               import inv
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar, FrequencyRadar, Track
from fdia_simulation.filters    import RadarFilterCA,RadarFilterCV,RadarFilterCT,RadarFilterTA
from fdia_simulation.filters    import MultipleRadarsFilterCA,MultipleRadarsFilterCV,MultipleRadarsFilterCT,MultipleRadarsFilterTA
from fdia_simulation.filters    import MultipleFreqRadarsFilterCA,MultipleFreqRadarsFilterCV,MultipleFreqRadarsFilterCT,MultipleFreqRadarsFilterTA
from fdia_simulation.benchmarks import Benchmark


class Benchmark1RadarTestEnv(ABC):

    @abstractmethod
    def setUp(self):
        pass

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

class Benchmark1RadarIMMTestEnv(Benchmark1RadarTestEnv):

    @abstractmethod
    def setUp(self):
        pass

    def test_initialization_is_imm(self):
        self.assertTrue(self.benchmark.filter_is_imm)

    def test_process_filter_computes_probs(self):
        self.benchmark.gen_data_set()
        self.benchmark.process_filter(with_nees = True)
        self.assertEqual(np.shape(self.benchmark.probs), (100,3))


class Benchmark1RadarCATestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
        # Filter definition: CA model
        self.radar_filter = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarCVTestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
        # Filter definition: CV model
        self.radar_filter = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarCTTestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
        # Filter definition: CT model
        self.radar_filter = RadarFilterCT(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)

class Benchmark1RadarTATestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
        # Filter definition: TA model
        self.radar_filter = RadarFilterTA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        # Benchmark definition
        self.benchmark   = Benchmark(radars = self.radar, radar_filter = self.radar_filter, states = self.states)


class Benchmark1RadarIMM2TestCase(Benchmark1RadarTestEnv,unittest.TestCase):
    def setUp(self):
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
        # Filter definition
        ## Classical models
        self.radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
        self.radar_filter_ta = RadarFilterTA(dim_x = 9, dim_z = 3, q = 100., radar = self.radar)
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
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
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
        # Radar definition
        self.radar = Radar(x=2000,y=2000)
        self.radar.step = 1.
        # States definition
        self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
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
#
#
# class Benchmark2RadarsTestCase(unittest.TestCase):
#     def setUp(self):
#         # Radar definition
#         self.radar1 = Radar(x=2000,y=2000)
#         self.radar1.step = 1
#         self.radar2 = Radar(x=1000,y=1000)
#         self.radar2.step = 1
#         self.radars = [self.radar1, self.radar2]
#         # States definition
#         self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
#         # Filter definition
#         ## Classical models
#         self.radar_filter_ca = MultipleRadarsFilterCA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_cv = MultipleRadarsFilterCV(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_ct = MultipleRadarsFilterCT(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_ta = MultipleRadarsFilterTA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         ## IMM with ca, cv and ct models
#         filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct]
#         mu = [0.33, 0.33, 0.33]
#         trans = np.array([[0.998, 0.001, 0.001],
#                           [0.050, 0.900, 0.050],
#                           [0.001, 0.001, 0.998]])
#         self.radar_filter_imm3 = IMMEstimator(filters = filters, mu = mu, M = trans)
#
#         # Benchmark definitions
#         self.benchmark_ca   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ca,   states = self.states)
#         self.benchmark_cv   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_cv,   states = self.states)
#         self.benchmark_ct   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ct,   states = self.states)
#         self.benchmark_ta   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ta,   states = self.states)
#         self.benchmark_imm3 = Benchmark(radars = self.radars, radar_filter = self.radar_filter_imm3, states = self.states)
#     # ==========================================================================
#     # ======================== Initialization tests ============================
#
#     def test_initialization_radars_2radars(self):
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ca.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_cv.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ct.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ta.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_imm3.radars)
#
#     def test_initialization_is_imm(self):
#         self.assertFalse(self.benchmark_ca.filter_is_imm)
#         self.assertFalse(self.benchmark_cv.filter_is_imm)
#         self.assertFalse(self.benchmark_ct.filter_is_imm)
#         self.assertFalse(self.benchmark_ta.filter_is_imm)
#         self.assertTrue(self.benchmark_imm3.filter_is_imm)
#
#     def test_initialization_measurement_sets(self):
#         self.assertFalse(self.benchmark_ca.measured_values)
#         self.assertFalse(self.benchmark_cv.measured_values)
#         self.assertFalse(self.benchmark_ct.measured_values)
#         self.assertFalse(self.benchmark_ta.measured_values)
#         self.assertFalse(self.benchmark_imm3.measured_values)
#         self.assertFalse(self.benchmark_ca.labeled_values)
#         self.assertFalse(self.benchmark_cv.labeled_values)
#         self.assertFalse(self.benchmark_ct.labeled_values)
#         self.assertFalse(self.benchmark_ta.labeled_values)
#         self.assertFalse(self.benchmark_imm3.labeled_values)
#
#     def test_initialization_pos_data(self):
#         pos_data = np.array([[i]*3 for i in range(100)])
#         self.assertTrue(np.array_equal(self.benchmark_ca.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_cv.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_ct.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_ta.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_imm3.pos_data,pos_data))
#
#     # ==========================================================================
#     # ========================== Function tests ================================
#
#     def test_gen_data_set(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.assertFalse(self.benchmark_ca.labeled_values)
#         self.assertFalse(self.benchmark_cv.labeled_values)
#         self.assertFalse(self.benchmark_ct.labeled_values)
#         self.assertFalse(self.benchmark_ta.labeled_values)
#         #self.assertFalse(self.benchmark_imm3.labeled_values)
#         self.assertEqual(np.shape(self.benchmark_ca.measured_values),(100,6))
#         self.assertEqual(np.shape(self.benchmark_cv.measured_values),(100,6))
#         self.assertEqual(np.shape(self.benchmark_ct.measured_values),(100,6))
#         self.assertEqual(np.shape(self.benchmark_ta.measured_values),(100,6))
#         #self.assertEqual(np.shape(self.benchmark_imm3.measured_values),(100,6))
#
#     def test_process_filter_nees_false(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = False)
#         self.benchmark_cv.process_filter(with_nees = False)
#         self.benchmark_ct.process_filter(with_nees = False)
#         self.benchmark_ta.process_filter(with_nees = False)
#         #self.benchmark_imm3.process_filter(with_nees = False)
#         self.assertTrue(self.benchmark_ca.nees.size == 0)
#         self.assertTrue(self.benchmark_cv.nees.size == 0)
#         self.assertTrue(self.benchmark_ct.nees.size == 0)
#         self.assertTrue(self.benchmark_ta.nees.size == 0)
#         #self.assertTrue(self.benchmark_imm3.nees.size == 0)
#
#     def test_process_filter_nees_true(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertEqual(np.shape(self.benchmark_ca.nees), (100,1))
#         self.assertEqual(np.shape(self.benchmark_cv.nees), (100,1))
#         self.assertEqual(np.shape(self.benchmark_ct.nees), (100,1))
#         self.assertEqual(np.shape(self.benchmark_ta.nees), (100,1))
#         #self.assertEqual(np.shape(self.benchmark_imm3.nees), (100,1))
#
#     def test_process_filter_computes_probs(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertTrue(self.benchmark_ca.probs.size == 0)
#         self.assertTrue(self.benchmark_cv.probs.size == 0)
#         self.assertTrue(self.benchmark_ct.probs.size == 0)
#         self.assertTrue(self.benchmark_ta.probs.size == 0)
#         #self.assertEqual(np.shape(self.benchmark_imm3.probs), (100,3))
#
#     def test_process_filter_correct_estimated_positions(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertEqual(np.shape(self.benchmark_ca.estimated_positions), (100,3))
#         self.assertEqual(np.shape(self.benchmark_cv.estimated_positions), (100,3))
#         self.assertEqual(np.shape(self.benchmark_ct.estimated_positions), (100,3))
#         self.assertEqual(np.shape(self.benchmark_ta.estimated_positions), (100,3))
#         #self.assertEqual(np.shape(self.benchmark_imm3.estimated_positions), (100,3))
#
#
# class Benchmark2FreqRadarsTestCase(unittest.TestCase):
#     def setUp(self):
#         # Radar definition
#         self.radar1 = FrequencyRadar(x=2000,y=2000)
#         self.radar2 = FrequencyRadar(x=1000,y=1000)
#         self.radars = [self.radar1, self.radar2]
#         # States definition
#         self.states = np.array([[i,i/2,i/10]*3 for i in range(100)])
#         # Filter definition
#         ## Classical models
#         self.radar_filter_ca = MultipleFreqRadarsFilterCA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_cv = MultipleFreqRadarsFilterCV(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_ct = MultipleFreqRadarsFilterCT(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         self.radar_filter_ta = MultipleFreqRadarsFilterTA(dim_x = 9, dim_z = 6, q = 100., radars = self.radars)
#         ## IMM with ca, cv and ct models
#         filters = [self.radar_filter_cv, self.radar_filter_ca, self.radar_filter_ct]
#         mu = [0.33, 0.33, 0.33]
#         trans = np.array([[0.998, 0.001, 0.001],
#                           [0.050, 0.900, 0.050],
#                           [0.001, 0.001, 0.998]])
#         self.radar_filter_imm3 = IMMEstimator(filters = filters, mu = mu, M = trans)
#
#         # Benchmark definitions
#         self.benchmark_ca   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ca,   states = self.states)
#         self.benchmark_cv   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_cv,   states = self.states)
#         self.benchmark_ct   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ct,   states = self.states)
#         self.benchmark_ta   = Benchmark(radars = self.radars, radar_filter = self.radar_filter_ta,   states = self.states)
#         self.benchmark_imm3 = Benchmark(radars = self.radars, radar_filter = self.radar_filter_imm3, states = self.states)
#     # ==========================================================================
#     # ======================== Initialization tests ============================
#
#     def test_initialization_radars_2radars(self):
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ca.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_cv.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ct.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_ta.radars)
#         self.assertEqual([self.radar1, self.radar2],self.benchmark_imm3.radars)
#
#     def test_initialization_is_imm(self):
#         self.assertFalse(self.benchmark_ca.filter_is_imm)
#         self.assertFalse(self.benchmark_cv.filter_is_imm)
#         self.assertFalse(self.benchmark_ct.filter_is_imm)
#         self.assertFalse(self.benchmark_ta.filter_is_imm)
#         self.assertTrue(self.benchmark_imm3.filter_is_imm)
#
#     def test_initialization_measurement_sets(self):
#         self.assertFalse(self.benchmark_ca.measured_values)
#         self.assertFalse(self.benchmark_cv.measured_values)
#         self.assertFalse(self.benchmark_ct.measured_values)
#         self.assertFalse(self.benchmark_ta.measured_values)
#         self.assertFalse(self.benchmark_imm3.measured_values)
#         self.assertFalse(self.benchmark_ca.labeled_values)
#         self.assertFalse(self.benchmark_cv.labeled_values)
#         self.assertFalse(self.benchmark_ct.labeled_values)
#         self.assertFalse(self.benchmark_ta.labeled_values)
#         self.assertFalse(self.benchmark_imm3.labeled_values)
#
#     def test_initialization_pos_data(self):
#         pos_data = np.array([[i]*3 for i in range(100)])
#         self.assertTrue(np.array_equal(self.benchmark_ca.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_cv.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_ct.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_ta.pos_data,pos_data))
#         self.assertTrue(np.array_equal(self.benchmark_imm3.pos_data,pos_data))
#
#     # ==========================================================================
#     # ========================== Function tests ================================
#
#     def test_gen_data_set(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.assertEqual(self.benchmark_ca.measured_values.size, 0)
#         self.assertEqual(self.benchmark_cv.measured_values.size, 0)
#         self.assertEqual(self.benchmark_ct.measured_values.size, 0)
#         self.assertEqual(self.benchmark_ta.measured_values.size, 0)
#         self.assertEqual(self.benchmark_imm3.measured_values.size, 0)
#         self.assertEqual(len(self.benchmark_ca.labeled_values),200)
#         self.assertEqual(len(self.benchmark_cv.labeled_values),200)
#         self.assertEqual(len(self.benchmark_ct.labeled_values),200)
#         self.assertEqual(len(self.benchmark_ta.labeled_values),200)
#         self.assertEqual(len(self.benchmark_imm3.labeled_values),200)
#
#     def test_process_filter_nees_false(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = False)
#         self.benchmark_cv.process_filter(with_nees = False)
#         self.benchmark_ct.process_filter(with_nees = False)
#         self.benchmark_ta.process_filter(with_nees = False)
#         #self.benchmark_imm3.process_filter(with_nees = False)
#         self.assertTrue(self.benchmark_ca.nees.size == 0)
#         self.assertTrue(self.benchmark_cv.nees.size == 0)
#         self.assertTrue(self.benchmark_ct.nees.size == 0)
#         self.assertTrue(self.benchmark_ta.nees.size == 0)
#         #self.assertTrue(self.benchmark_imm3.nees.size == 0)
#
#     def test_process_filter_nees_true(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertEqual(np.shape(self.benchmark_ca.nees), (200,1))
#         self.assertEqual(np.shape(self.benchmark_cv.nees), (200,1))
#         self.assertEqual(np.shape(self.benchmark_ct.nees), (200,1))
#         self.assertEqual(np.shape(self.benchmark_ta.nees), (200,1))
#         #self.assertEqual(np.shape(self.benchmark_imm3.nees), (100,1))
#
#     def test_process_filter_computes_probs(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertTrue(self.benchmark_ca.probs.size == 0)
#         self.assertTrue(self.benchmark_cv.probs.size == 0)
#         self.assertTrue(self.benchmark_ct.probs.size == 0)
#         self.assertTrue(self.benchmark_ta.probs.size == 0)
#         #self.assertEqual(np.shape(self.benchmark_imm3.probs), (100,3))
#
#     def test_process_filter_correct_estimated_positions(self):
#         self.benchmark_ca.gen_data_set()
#         self.benchmark_cv.gen_data_set()
#         self.benchmark_ct.gen_data_set()
#         self.benchmark_ta.gen_data_set()
#         self.benchmark_imm3.gen_data_set()
#         self.benchmark_ca.process_filter(with_nees = True)
#         self.benchmark_cv.process_filter(with_nees = True)
#         self.benchmark_ct.process_filter(with_nees = True)
#         self.benchmark_ta.process_filter(with_nees = True)
#         #self.benchmark_imm3.process_filter(with_nees = True)
#         self.assertEqual(np.shape(self.benchmark_ca.estimated_positions), (200,3))
#         self.assertEqual(np.shape(self.benchmark_cv.estimated_positions), (200,3))
#         self.assertEqual(np.shape(self.benchmark_ct.estimated_positions), (200,3))
#         self.assertEqual(np.shape(self.benchmark_ta.estimated_positions), (200,3))
#         #self.assertEqual(np.shape(self.benchmark_imm3.estimated_positions), (100,3))
