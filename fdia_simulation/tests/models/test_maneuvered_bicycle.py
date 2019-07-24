# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:32:12 2019

@author: qde
"""

import unittest
from fdia_simulation.models import Command, ManeuveredBicycle


class BicycleTestCase(unittest.TestCase):
    def setUp(self):
        self.head_cmd_test  = Command('head',0,0,0)
        self.vel_cmd_test = Command('vel',0.3,0,0)
        self.bicycle_test = ManeuveredBicycle(x0 = 0, y0 = 0, v0 = 0.3, h0 = 0,
                                                       command_list = [self.head_cmd_test,
                                                                       self.vel_cmd_test])

    def test_initial_x(self):
        self.assertEqual(self.bicycle_test.x,0)

    def test_initial_y(self):
        self.assertEqual(self.bicycle_test.y,0)

    def test_initial_vel(self):
        self.assertEqual(self.bicycle_test.vel,0.3)

    def test_initial_head(self):
        self.assertEqual(self.bicycle_test.head,0)

    def test_initial_heading_command(self):
        self.assertEqual(self.head_cmd_test,Command('head',0,0,0))

    def test_initial_velocity_command(self):
        self.assertEqual(self.vel_cmd_test,Command('vel',0.3,0,0))

    def test_initial_heading_command_in_commands(self):
        self.assertEqual(self.bicycle_test.commands['head'],Command('head',0,0,0))

    def test_initial_velocity_command_in_commands(self):
        self.assertEqual(self.bicycle_test.commands['vel'],Command('vel',0.3,0,0))

    def test_change_heading_command(self):
        self.bicycle_test.change_head(10,15)
        cmd_head = self.bicycle_test.commands['head']
        self.assertEqual(cmd_head.value,10)
        self.assertEqual(cmd_head.steps,15)

    def test_change_velocity_command(self):
        self.bicycle_test.change_vel(1,15)
        cmd_vel  = self.bicycle_test.commands['vel']
        self.assertEqual(cmd_vel.value,1)
        self.assertEqual(cmd_vel.steps,15)

    def test_change_command_with_heading(self):
        self.bicycle_test.change_command('head',15,20)
        cmd_head = self.bicycle_test.commands['head']
        self.assertEqual(cmd_head.value,15)
        self.assertEqual(cmd_head.steps,20)
        delta = 15/20
        self.assertEqual(cmd_head.delta,delta)

    def test_change_command_with_velocity(self):
        self.bicycle_test.change_command('vel',5,20)
        cmd_vel  = self.bicycle_test.commands['vel']
        self.assertEqual(cmd_vel.value,5)
        self.assertEqual(cmd_vel.steps,20)
        delta = (5 - 0.3) / 20
        self.assertEqual(cmd_vel.delta,delta)


    def test_change_velocity_command_effect(self):
        self.bicycle_test.change_command('vel',5.,20)
        cmd_vel = self.bicycle_test.commands['vel']
        for _ in range(20):
            self.bicycle_test.update()
        self.assertAlmostEqual(self.bicycle_test.vel,5.) # 5.+1e-15
        self.assertEqual(cmd_vel.value,5.)
        self.assertEqual(cmd_vel.steps,0.)

    def test_change_head_command_effect(self):
        self.bicycle_test.change_command('head',5.,20)
        cmd_head = self.bicycle_test.commands['head']
        for _ in range(20):
            self.bicycle_test.update()
        self.assertEqual(self.bicycle_test.head,5.)
        self.assertEqual(cmd_head.value,5.)
        self.assertEqual(cmd_head.steps,0.)


if __name__ == "__main__":
    unittest.main()
