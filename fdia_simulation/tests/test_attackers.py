# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:15:55 2019

@author: qde
"""

import unittest
import numpy as np
from nose.tools                import raises
from numpy.random              import randn
from filterpy.kalman           import KalmanFilter
from fdia_simulation.attackers import MoAttacker, Attacker

class AttackerTestCase(unittest.TestCase):

    @raises(TypeError)
    def test_no_initialization(self):
        abstractClassInstance = Attacker()
        abstractClassInstance.change_measurements()

class MoAttackerTestCase(unittest.TestCase):
    def setUp(self):
        kf = KalmanFilter(dim_x=2,dim_z=1)
        kf.F = np.array([[1., 1.],
                         [0., 1.]])
        kf.H = np.eye(2)
        kf.Q = np.eye(2)
        kf.R = np.eye(2)
        kf.x = np.array([[0.,0.]]).T
        self.kf = kf
        self.attacker = MoAttacker(self.kf)
        self.data_count = 100
        self.zs = [(i+randn()) for i in range(self.data_count)]

    def test_add_false_measurement(self):
        self.attacker.attack_sequence = np.zeros((2,0))
        self.attacker.add_false_measurement(np.array([[1.,1.]]).T)
        self.assertTrue(np.array_equal(self.attacker.attack_sequence,np.array([[1.],[1.]])))

    def test_no_unstable_values(self):
        A = np.diag((-1,-2,-3))
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(self.attacker.unst_data),0)

    def test_eigenvalues_found_and_added(self):
        A = np.diag((1,2,3))
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(self.attacker.unst_data),3)
        self.assertEqual(self.attacker.unst_data[0].value,1)
        self.assertEqual(self.attacker.unst_data[1].value,2)
        self.assertEqual(self.attacker.unst_data[2].value,3)

    def test_eigenvectors_found_and_added(self):
        A = np.diag((1,2,3))
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(self.attacker.unst_data),3)
        self.assertTrue(np.array_equal(self.attacker.unst_data[0].vector,np.array([1,0,0])))
        self.assertTrue(np.array_equal(self.attacker.unst_data[1].vector,np.array([0,1,0])))
        self.assertTrue(np.array_equal(self.attacker.unst_data[2].vector,np.array([0,0,1])))

if __name__ == "__main__":
    unittest.main()
