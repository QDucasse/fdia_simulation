# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:38:04 2019

@author: qde
"""

import unittest
from fdia_simulation.models import Command, ManeuveredAircraft

class AircraftTestCase(unittest.TestCase):
    def setUp(self):
        self.headx_cmd_test  = Command('headx',0,0,0)
        self.headz_cmd_test  = Command('headz',0,0,0)
        self.vel_cmd_test = Command('vel',0.3,0,0)
        self.aircraft_test = ManeuveredAircraft(v0 = 0.3,
                                                command_list = [self.headx_cmd_test,
                                                                self.headz_cmd_test,
                                                                self.vel_cmd_test])

    def test_initial_x(self):
        self.assertEqual(self.aircraft_test.x,0)

    def test_initial_y(self):
        self.assertEqual(self.aircraft_test.y,0)

    def test_initial_z(self):
        self.assertEqual(self.aircraft_test.z,0)

    def test_initial_vel(self):
        self.assertEqual(self.aircraft_test.vel,0.3)

    def test_initial_headx(self):
        self.assertEqual(self.aircraft_test.headx,0)

    def test_initial_headz(self):
        self.assertEqual(self.aircraft_test.headz,0)

    def test_initial_headingx_command(self):
        self.assertEqual(self.headx_cmd_test,Command('headx',0,0,0))

    def test_initial_headingz_command(self):
        self.assertEqual(self.headz_cmd_test,Command('headz',0,0,0))

    def test_initial_velocity_command(self):
        self.assertEqual(self.vel_cmd_test,Command('vel',0.3,0,0))

    def test_initial_headingx_command_in_commands(self):
        self.assertEqual(self.aircraft_test.commands['headx'],Command('headx',0,0,0))

    def test_initial_headingz_command_in_commands(self):
        self.assertEqual(self.aircraft_test.commands['headz'],Command('headz',0,0,0))

    def test_initial_velocity_command_in_commands(self):
        self.assertEqual(self.aircraft_test.commands['vel'],Command('vel',0.3,0,0))

    def test_change_headingx_command(self):
        self.aircraft_test.change_headx(10,15)
        cmd_headx = self.aircraft_test.commands['headx']
        self.assertEqual(cmd_headx.value,10)
        self.assertEqual(cmd_headx.steps,15)

    def test_change_headingz_command(self):
        self.aircraft_test.change_headz(10,15)
        cmd_headz = self.aircraft_test.commands['headz']
        self.assertEqual(cmd_headz.value,10)
        self.assertEqual(cmd_headz.steps,15)

    def test_change_velocity_command(self):
        self.aircraft_test.change_vel(1,15)
        cmd_vel  = self.aircraft_test.commands['vel']
        self.assertEqual(cmd_vel.value,1)
        self.assertEqual(cmd_vel.steps,15)

    def test_change_command_with_headingx(self):
        self.aircraft_test.change_command('headx',15,20)
        cmd_headx = self.aircraft_test.commands['headx']
        self.assertEqual(cmd_headx.value,15)
        self.assertEqual(cmd_headx.steps,20)
        self.assertEqual(cmd_headx.delta,15/20)

    def test_change_command_with_headingz(self):
        self.aircraft_test.change_command('headz',15,20)
        cmd_headz = self.aircraft_test.commands['headz']
        self.assertEqual(cmd_headz.value,15)
        self.assertEqual(cmd_headz.steps,20)
        self.assertEqual(cmd_headz.delta,15/20)

    def test_change_command_with_velocity(self):
        self.aircraft_test.change_command('vel',5,20)
        cmd_vel  = self.aircraft_test.commands['vel']
        self.assertEqual(cmd_vel.value,5)
        self.assertEqual(cmd_vel.steps,20)

    def test_change_headx_command_effect(self):
        self.aircraft_test.change_command('headx',5.,20)
        cmd_headx = self.aircraft_test.commands['headx']
        for _ in range(20):
            self.aircraft_test.update()
        self.assertEqual(self.aircraft_test.headx,5.)
        self.assertEqual(cmd_headx.value,5.)
        self.assertEqual(cmd_headx.steps,0.)

    def test_change_headz_command_effect(self):
        self.aircraft_test.change_command('headz',5.,20)
        cmd_headz = self.aircraft_test.commands['headz']
        for _ in range(20):
            self.aircraft_test.update()
        self.assertEqual(self.aircraft_test.headz,5.)
        self.assertEqual(cmd_headz.value,5.)
        self.assertEqual(cmd_headz.steps,0.)

if __name__ == "__main__":
    unittest.main()
