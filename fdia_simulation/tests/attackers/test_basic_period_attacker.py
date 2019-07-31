# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:37:53 2019

@author: qde
"""

import unittest
import numpy as np
from nose.tools                import raises
from filterpy.kalman           import KalmanFilter,ExtendedKalmanFilter
from fdia_simulation.models    import Radar, PeriodRadar, LabeledMeasurement
from fdia_simulation.attackers import PeriodAttacker, BruteForcePeriodAttacker, DriftPeriodAttacker

class PeriodAttackerTestCase(unittest.TestCase):
    def setUp(self):
        # Simulated filter for 2 radars (2*3 measurements)
        self.filter = ExtendedKalmanFilter(dim_x = 9, dim_z = 6)
        self.gamma = np.eye(3)
        self.mag_vector = np.array([[-10, -10, -10]]).T
        self.t0 = 10
        self.time = 50
        self.radar = Radar(x=10,y=10)
        self.radar_pos = 1
        self.attacker = PeriodAttacker(filter = self.filter,
                                          radar = self.radar, radar_pos = self.radar_pos,
                                          gamma = self.gamma,mag_vector = self.mag_vector,
                                          t0 = self.t0, time = self.time)

    # ==========================================================================
    # ========================== Initialization tests ==========================

    def test_initialization_no_errors(self):
        self.assertTrue(np.array_equal(self.attacker.gamma,self.gamma))
        self.assertTrue(np.array_equal(self.attacker.mag_vector,self.mag_vector))
        self.assertEqual(self.t0,10)
        self.assertEqual(self.time,50)

    def test_initialization_wrong_mag_vector(self):
        with self.assertRaises(ValueError):
            mag_vector = np.array([[-10,-10]])
            attacker = PeriodAttacker(filter = self.filter,
                                         radar = self.radar, radar_pos = self.radar_pos,
                                         gamma = self.gamma,mag_vector = mag_vector,
                                         t0 = self.t0, time = self.time)

    def test_initialization_wrong_gamma(self):
        with self.assertRaises(ValueError):
            filter = ExtendedKalmanFilter(dim_x = 9, dim_z = 6)
            gamma  = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
            attacker = PeriodAttacker(filter = self.filter,
                                         radar = self.radar, radar_pos = self.radar_pos,
                                         gamma = gamma,mag_vector = self.mag_vector,
                                         t0 = self.t0, time = self.time)

    def test_initialization_attack_no_effect(self):
        with self.assertWarns(Warning):
            mag_vector = np.array([[0, 0, 0]]).T
            att  =  PeriodAttacker(filter = self.filter,
                                      radar = self.radar, radar_pos = self.radar_pos,
                                      gamma = self.gamma, mag_vector = mag_vector,
                                      t0 = self.t0, time = self.time)

    def test_initialization_given_pos(self):
        # Expected results
        gamma =  np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        mag_vector = np.array([[1,1,1]]).T
        radar_pos = 0

        #Generation
        att = PeriodAttacker(filter = self.filter, radar_pos = radar_pos,
                                radar = self.radar,t0 = self.t0, time = self.time)
        computed_gamma      = att.gamma
        computed_mag_vector = att.mag_vector

        # Comparison
        self.assertTrue(np.array_equal(gamma,computed_gamma))
        self.assertTrue(np.array_equal(mag_vector,computed_mag_vector))
    # =======================================================
    # ========================== Listening/Attack tests ========================

    def test_listen_measurement_increments_time(self):
        measurements = [np.ones((3,1))*i for i in range(100)]
        tags  = [1]*100
        times = [i/10 for i in range(100)]
        labeled_measurements = [LabeledMeasurement(tag,time,value) for tag,time,value in zip(tags,times,measurements)]
        for i,labeled_measurement in enumerate(labeled_measurements):
            self.assertEqual(self.attacker.current_time,i)
            self.attacker.listen_measurement(labeled_measurement)

    def test_listen_measurement_1_step_attack(self):
        measurement = np.array([[10,10,10]]).T
        tag         = 1
        time        = 1
        labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = measurement)

        modified_measurement = np.array([[0,0,0]]).T
        tag         = 1
        time        = 1
        modified_labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = modified_measurement)

        self.attacker.t0     = 0
        computed_measurement = self.attacker.listen_measurement(labeled_measurement)
        self.assertEqual(modified_labeled_measurement,computed_measurement)

    def test_listen_measurement_1_step_no_attack(self):
        measurement = np.array([[10,10,10]]).T
        tag         = 1
        time        = 1
        labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = measurement)
        computed_measurement = self.attacker.listen_measurement(labeled_measurement)
        self.assertEqual(labeled_measurement,computed_measurement)

    def test_unattacked_vectors(self):
        measurements = [np.ones((3,1))*i for i in range(100)]
        tags  = [1]*100
        times = [i/10 for i in range(100)]
        labeled_measurements = [LabeledMeasurement(tag,time,value) for tag,time,value in zip(tags,times,measurements)]

        modified_measurements = []
        for i,labeled_measurement in enumerate(labeled_measurements):
            mod_meas = self.attacker.listen_measurement(labeled_measurement)
            modified_measurements.append(mod_meas)

        # Unattacked measurements from 0 to 10
        measurements = [np.ones((3,1))*i for i in range(10)]
        tags  = [1]*10
        times = [i/10 for i in range(10)]
        unattacked_measurements1 = [LabeledMeasurement(tag,time,value) for tag,time,value in zip(tags,times,measurements)]

        # Unattacked measurements from 60 to 100
        measurements = [np.ones((3,1))*i for i in range(60,100)]
        tags  = [1]*40
        times = [i/10 for i in range(60,100)]
        unattacked_measurements2 = [LabeledMeasurement(tag,time,value) for tag,time,value in zip(tags,times,measurements)]

        comparison_list_1 = zip(unattacked_measurements1,  modified_measurements[0:10])
        comparison_list_2 = zip(unattacked_measurements2,  modified_measurements[60:100])
        self.assertTrue(all((meas == mod_meas) for meas, mod_meas in comparison_list_1))
        self.assertTrue(all((meas == mod_meas) for meas, mod_meas in comparison_list_2))

    # def test_attacked_vectors(self):
    #     measurements = [np.ones((3,1))*i for i in range(100)]
    #     tags  = [1]*100
    #     times = [i/10 for i in range(100)]
    #     labeled_measurements = [LabeledMeasurement(tag,time,value) for tag,time,value in zip(tags,times,measurements)]
    #
    #     modified_measurements = []
    #     for i,labeled_measurement in enumerate(labeled_measurements):
    #         mod_meas = self.attacker.listen_measurement(labeled_measurement)
    #         modified_measurements.append(mod_meas)
    #
    #     attacked_meas = [np.subtract(np.array([[i,i,i,i,i,i]]).T,np.array([[0,0,0,10,10,10]]).T) for i in range(10,60)]
    #     comparison_list = zip(attacked_meas,  modified_measurements[10:60])
    #     self.assertTrue(all([np.allclose(meas, mod_meas) for meas, mod_meas in comparison_list]))


class BruteForcePeriodAttackerTestCase(PeriodAttackerTestCase):
    def setUp(self):
        PeriodAttackerTestCase.setUp(self)
        self.mag = 1e6
        self.attacker = BruteForcePeriodAttacker(filter = self.filter, mag = self.mag,
                                           gamma = self.gamma,mag_vector = self.mag_vector,
                                           radar_pos = self.radar_pos, radar = self.radar,
                                           t0 = self.t0, time = self.time)

    def test_initialization_no_errors(self):
        self.assertTrue(np.array_equal(self.attacker.gamma,self.gamma))
        self.assertTrue(np.array_equal(self.attacker.mag_vector,self.mag_vector*self.mag))
        self.assertEqual(self.t0,10)
        self.assertEqual(self.time,50)

    def test_listen_measurement_1_step_attack(self):
        measurement          = np.array([[10,10,10]]).T
        tag         = 1
        time        = 1
        labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = measurement)

        modified_measurement = np.array([[-9.99999e6,-9.99999e6,-9.99999e6]]).T
        tag         = 1
        time        = 1
        modified_labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = modified_measurement)



        self.attacker.t0 = 0
        computed_measurement = self.attacker.listen_measurement(labeled_measurement)
        self.assertEqual(modified_labeled_measurement,computed_measurement)

#     def test_attacked_vectors(self):
#         pass

class DriftPeriodAttackerTestCase(PeriodAttackerTestCase):
    def setUp(self):
        self.radar_position = 1
        PeriodAttackerTestCase.setUp(self)
        self.attacker = DriftPeriodAttacker(filter = self.filter,
                                      radar = self.radar, radar_pos = self.radar_position,
                                      mag_vector = self.mag_vector,
                                      t0 = self.t0, time = self.time)

    def test_initialization_no_errors(self):
        self.assertTrue(np.array_equal(self.attacker.gamma,self.gamma))
        self.assertTrue(np.array_equal(self.attacker.mag_vector,self.mag_vector))
        self.assertEqual(self.t0,10)
        self.assertEqual(self.time,50)
        self.assertEqual(self.radar, Radar(x=10,y=10))
        self.assertTrue(np.array_equal(self.attacker.attack_drift,np.array([[0,0,10]]).T))

    def test_listen_measurement_1_step_attack(self):
        measurement          = np.array([[10,10,10]]).T
        tag         = 1
        time        = 1
        labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = measurement)

        modified_measurement = np.array([[9.,0.,0.]]).T
        tag         = 1
        time        = 1
        modified_labeled_measurement = LabeledMeasurement(tag = tag, time = time, value = modified_measurement)

        self.attacker.t0 = 0
        computed_measurement = self.attacker.listen_measurement(labeled_measurement)
        print(computed_measurement)
        self.assertEqual(modified_labeled_measurement,computed_measurement)

#     def test_attacked_vectors(self):
#         pass


if __name__ == "__main__":
    unittest.main()
