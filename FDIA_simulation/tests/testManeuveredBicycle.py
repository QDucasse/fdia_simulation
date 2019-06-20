# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:32:12 2019

@author: qde
"""

import unittest
from movingTarget import Command
from maneuveredBicycle import ManeuveredBicycle

class ManeuveredBicycleTestCase(unittest.TestCase):
    def setUp(self):
        self.headingCommandTest  = Command('head',0,0,0)
        self.velocityCommandTest = Command('vel',0.3,0,0)
        self.maneuveredBicycleTest = ManeuveredBicycle(x0 = 0, y0 = 0, v0 = 0.3, h0 = 0,
                                                       command_list = [self.headingCommandTest,
                                                                       self.velocityCommandTest])
    
    def test_initial_x(self):
        self.assertEqual(self.maneuveredBicycleTest.x,0)

    def test_initial_y(self):
        self.assertEqual(self.maneuveredBicycleTest.y,0)
        
    def test_initial_vel(self):
        self.assertEqual(self.maneuveredBicycleTest.vel,0.3)
        
    def test_initial_head(self):
        self.assertEqual(self.maneuveredBicycleTest.head,0)
        
    def test_initial_heading_command(self):
        self.assertEqual(self.headingCommandTest,Command('head',0,0,0))
        
    def test_initial_velocity_command(self):
        self.assertEqual(self.velocityCommandTest,Command('vel',0.3,0,0))
        
    def test_initial_heading_command_in_commands(self):
        self.assertEqual(self.maneuveredBicycleTest.commands['head'],Command('head',0,0,0))
        
    def test_initial_velocity_command_in_commands(self):
        self.assertEqual(self.maneuveredBicycleTest.commands['vel'],Command('vel',0.3,0,0))
    
    def test_change_heading_command(self):
        self.maneuveredBicycleTest.change_head(10,15)
        cmd_head = self.maneuveredBicycleTest.commands['head']
        self.assertEqual(cmd_head.value,10)
        self.assertEqual(cmd_head.steps,15)
        
    def test_change_velocity_command(self):
        self.maneuveredBicycleTest.change_vel(1,15)
        cmd_vel  = self.maneuveredBicycleTest.commands['vel']
        self.assertEqual(cmd_vel.value,1)
        self.assertEqual(cmd_vel.steps,15)
        
    def test_change_command_with_heading(self):
        self.maneuveredBicycleTest.change_command('head',13,17)
        cmd_head = self.maneuveredBicycleTest.commands['head']
        self.assertEqual(cmd_head.value,13)
        self.assertEqual(cmd_head.steps,17)
        
    def test_change_command_with_velocity(self):
        self.maneuveredBicycleTest.change_command('vel',5,20)
        cmd_vel  = self.maneuveredBicycleTest.commands['vel']
        self.assertEqual(cmd_vel.value,5)
        self.assertEqual(cmd_vel.steps,20)
        
if __name__ == "__main__":
    unittest.main()

