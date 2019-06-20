# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:28:45 2019

@author: qde
"""

import unittest
from fault_detectors.fault_detector import FaultDetector,ChiSquareDetector,EuclidianDetector
from filterpy.common import kinematic_kf
from numpy.random import randn


class FaultDetectorTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_no_initialization(self):
        abstractClassInstance = FaultDetector()

class ChiSquareDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.kinematic_test_kf = kinematic_kf(dim=1,order=1,dt=1)
        x         = [0.,2.]
        self.zs   = [x[0]]
        self.pos  = [x[0]]
        noise_std = 1.
        
        # Noisy measurements generation for 30 samples
        for _ in range(30): 
            last_pos = x[0]
            last_vel = x[1]
            new_vel  = last_vel
            new_pos  = last_pos + last_vel
            x = [new_pos, new_vel]
            z = new_pos + (randn()*noise_std)
            self.zs.append(z)
        
        # Outlier generation
        self.zs[5]  += 10.
        self.zs[10] += 10.
        self.zs[15] += 10.
        self.zs[20] += 10.
        self.zs[25] += 10.
        
        self.detector = ChiSquareDetector()
        
    def test_initial_reviewed_values(self):
        self.assertEqual(self.detector.reviewed_values,[])
        
    def test_correct_value_not_detected(self):
        self.kinematic_test_kf.predict()
        result = self.detector.review_measurement(self.zs[0],self.kinematic_test_kf)
        self.assertEqual(result,"Success")
        
    def test_correct_value_added_to_reviewed_list(self):
        self.kinematic_test_kf.predict()
        self.assertEqual(0, len(self.detector.reviewed_values))
        self.detector.review_measurement(self.zs[0],self.kinematic_test_kf)
        self.assertEqual(1, len(self.detector.reviewed_values))
        
    def test_probabilities_added_to_reviewed_list(self):
        for z in self.zs:
            self.kinematic_test_kf.predict()
            self.detector.review_measurement(z,self.kinematic_test_kf)
            self.kinematic_test_kf.update(z)
        self.assertEqual(31,len(self.detector.reviewed_values))
        
    def test_wrong_value_detected(self):
        # First correct values
        for i in range(5):
            self.kinematic_test_kf.predict()
            self.detector.review_measurement(self.zs[i],self.kinematic_test_kf)
            self.kinematic_test_kf.update(self.zs[i])
        
        # Incorrect value
        self.kinematic_test_kf.predict()
        result = self.detector.review_measurement(self.zs[5],self.kinematic_test_kf)
        self.kinematic_test_kf.update(self.zs[5])
        self.assertEqual(result,"Failure")
        

class EuclidianDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.kinematic_test_kf = kinematic_kf(dim=1,order=1,dt=1)
        self.zs = []
        x = [0.,6.]
        noise_std = 0.001
        
        # Noisy measurements generation for 30 samples
        for _ in range(30): 
            last_pos = x[0]
            last_vel = x[1]
            new_vel  = last_vel
            new_pos  = last_pos + last_vel
            x = [new_pos, new_vel]
            z = new_pos + (randn()*noise_std)
            self.zs.append(z)
        
        # Outlier generation
        self.zs[5]  += 10.
        self.zs[10] += 10.
        self.zs[15] += 10.
        self.zs[20] += 10.
        self.zs[25] += 10.
        
        self.detector = EuclidianDetector()


if __name__ == "__main__":
    unittest.main()


