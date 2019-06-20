# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:30:25 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,radians
from fdia_simulation.models.moving_target import MovingTarget, Command, NoisySensor
from fdia_simulation.helpers.plotting import plot_measurements

def angle_between(x, y):
    '''
    Helper function computing the angle between two angles in degrees.

    Parameters
    ----------
    x : int
        First angle.

    y : int
        Second angle.

    Returns
    -------
    angle : int
        Angle between x and y taking in consideration the modulo 360.
    '''
    return min(y-x, y-x+360, y-x-360, key=abs)

class ManeuveredBicycle(MovingTarget):

    def __init__(self, x0, y0 ,v0, h0, command_list):
        super().__init__(command_list)
        self.x        = x0 # Position following x-axis
        self.y        = y0 # Position following y-axis
        self.vel      = v0 # Velocity
        self.head     = h0 # Heading of the front wheel


    def update(self):
        '''
        Updates the position of the model with respect to the commands. Returns
        the position and alters the model in consequence.

        Returns
        -------
        x : float
            New position following x-axis.

        y : float
            New position following y-axis.
        '''
        cmd_head = self.commands['head']
        cmd_vel  = self.commands['vel']
        velx     = self.vel * cos(radians(90-self.head))
        vely     = self.vel * sin(radians(90-self.head))
        self.x   += velx # Model application: x = x + dt*velx   (dt = 1 second)
        self.y   += vely # Model application: y = y + dt*vely   (dt = 1 second)

        if cmd_head.steps > 0:          # Heading command update
            cmd_head.steps -= 1         # Diminution of the number of steps
            self.head += cmd_head.delta # Adding a delta per step

        if cmd_vel.steps > 0:           # Velocity command update
            cmd_vel.steps -= 1          # Diminution of the number of steps
            self.vel += cmd_vel.delta   # Adding a delta per step

        return (self.x, self.y)

    def change_head(self, hdg_degrees, steps):
        '''
        Changes the heading command.

        Parameters
        ----------
        hdg_degrees: int
            New objective heading (in degrees).

        steps: int
            Number of steps within which the new heading value should be reached.

        Notes
        -----
        The change function should be called through the change_command function
        defined under the MovingTarget abstract class.
        '''
        cmd_head = self.commands['head']
        cmd_head.value = hdg_degrees
        cmd_head.steps = steps
        cmd_head.delta = angle_between(cmd_head.value, self.head) / steps
        if abs(cmd_head.delta) > 0:
            cmd_head.steps = steps
        else:
            cmd_head.steps = 0

    def change_vel(self, speed, steps):
        '''
        Changes the velocity command.

        Parameters
        ----------
        speed: float
            New objective speed (in m/s).

        steps: int
            Number of steps within which the new speed value should be reached.

        Notes
        -----
        The change function should be called through the change_command function
        defined under the MovingTarget abstract class.
        '''
        cmd_vel = self.commands['vel']
        cmd_vel.value = speed
        cmd_vel.steps = steps
        cmd_vel.delta = (cmd_vel.value - self.vel) / cmd_vel.steps
        if abs(cmd_vel.delta) > 0:
            cmd_vel.steps = steps
        else:
            cmd_vel.steps = 0


if __name__ == "__main__":

        # Route generation example with a ManeuveredBicycle
        sensor_std = 1.
        head_cmd = Command('head',0,0,0)
        vel_cmd  = Command('vel',0.3,0,0)
        bicycle  = ManeuveredBicycle(x0 = 0, y0 = 0, v0 = 0.3, h0 = 0, command_list = [head_cmd, vel_cmd])
        xs, ys = [], []

        # First phase
        for i in range(30):
            x, y = bicycle.update()
            xs.append(x)
            ys.append(y)

        # Change in commands
        bicycle.change_command("head",310, 25)
        bicycle.change_command("vel",1, 15)

        # Second phase
        for i in range(60):
            x, y = bicycle.update()
            xs.append(x)
            ys.append(y)

        # Sensor definitions: Position(x,y)
        nsx = NoisySensor(sensor_std)   # Position sensor for the x-axis
        nsy = NoisySensor(sensor_std)   # Position sensor for the y-axis
        pos = np.array(list(zip(xs, ys)))             # Real positions
        noisy_xs = [nsx.sense(x) for x in xs]
        noisy_ys = [nsy.sense(y) for y in ys]
        zs  = np.array(list(zip(noisy_xs, noisy_ys))) # Measured positions

        # Route plot
        plt.figure()
        plot_measurements(*zip(*zs), alpha=0.5)
        plt.plot(*zip(*pos), color='b', label='track')
        plt.axis('equal')
        plt.legend(loc=4)
        plt.grid(True)
