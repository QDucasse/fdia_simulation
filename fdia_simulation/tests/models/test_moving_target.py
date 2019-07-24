# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:14:47 2019

@author: qde
"""

import unittest
from nose.tools             import raises
from fdia_simulation.models import Command,MovingTarget

class CommandTestCase(unittest.TestCase):
    def setUp(self):
        self.commandTest = Command("velocity",0,1,2)

    def test_initial_name(self):
        self.assertEqual(self.commandTest.name,"velocity")

    def test_initial_value(self):
        self.assertEqual(self.commandTest.value,0)

    def test_initial_steps(self):
        self.assertEqual(self.commandTest.steps,1)

    def test_initial_delta(self):
        self.assertEqual(self.commandTest.delta,2)

    def test_command_equality(self):
        self.assertEqual(self.commandTest,Command("velocity",0,1,2))

    def test_command_repr(self):
        self.assertEqual(str(self.commandTest),"Command object\nname = velocity\nvalue = 0\nsteps = 1\ndelta = 2")


class MovingTargetTestCase(unittest.TestCase):
    @raises(TypeError)
    def test_no_initialization(self):
        abstractClassInstance = MovingTarget()


if __name__ == "__main__":
    unittest.main()
