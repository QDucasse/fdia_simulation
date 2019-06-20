# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:15:55 2019

@author: qde
"""

import unittest
import numpy as np
from fdia_simulation.attackers.attacker import Attacker, MoAttacker
from filterpy.common import kinematic_kf

class AttackerTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_no_initialization(self):
        abstractClassInstance = Attacker()
        abstractClassInstance.change_measurements()

class MoAttackerTestCase(unittest.TestCase):
    def setUp(self):
        self.kf = kinematic_kf(dim=1,order=1,dt=1)
        self.attacker = MoAttacker(self.kf)

    def test_add_false_measurement(self):
        self.attacker.attack_sequence = np.zeros((2,0))
        self.attacker.add_false_measurement(np.array([[1.,1.]]).T)
        self.assertTrue(np.array_equal(self.attacker.attack_sequence,np.array([[1.],[1.]])))

    def test_no_unstable_values(self):
        A = np.diag((-1,-2,-3))
        eig_dict = self.attacker.unst_data
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(eig_dict),0)

    def test_eigenvalues_found_and_added(self):
        A = np.diag((1,2,3))
        unst_data = self.attacker.unst_data
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(unst_data),3)
        self.assertEqual(unst_data[0].value,1)
        self.assertEqual(unst_data[1].value,2)
        self.assertEqual(unst_data[2].value,3)

    def test_eigenvectors_found_and_added(self):
        A = np.diag((1,2,3))
        unst_data = self.attacker.unst_data
        self.attacker.compute_unstable_eig(A)
        self.assertEqual(len(unst_data),3)
        self.assertTrue(np.array_equal(unst_data[0].vector,np.array([1,0,0])))
        self.assertTrue(np.array_equal(unst_data[1].vector,np.array([0,1,0])))
        self.assertTrue(np.array_equal(unst_data[2].vector,np.array([0,0,1])))

if __name__ == "__main__":

    unittest.main()
