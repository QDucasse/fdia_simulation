# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:30:25 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from math                   import cos,sin,radians
from mpl_toolkits.mplot3d   import Axes3D
from fdia_simulation.models import MovingTarget, Command, NoisySensor

class ManeuveredAircraft(MovingTarget):
    r'''Implements a model for a maneuvered aircraft: commands on two headings
    and velocity. The sensors are three position sensors.

    Parameters
    ----------
    x0, y0, z0: floats
        Initial positions along x-axis, y-axis and z-axis.

    v0: float
        Initial velocity of the system.

    hz0: int
        Initial heading of the system around z-axis (left-right turns).

    hy0: int
        Initial heading of the system around x-axis (up-down turns).

    command_list: Command iterable
        List of the Command objects refering to the commanded variables in our system.

    Attributes
    ----------
    x, y, z: floats
        Positions of the system along x-axis, y-axis and z-axis.

    vel: float
        Velocity of the system.

    headz: int
        Heading of the system around z-axis.

    headx: int
        Heading of the system around x-axis.
    '''
    def __init__(self, x0 = 0, y0 = 0, z0 = 0, v0 = 0, hz0 = 0, hx0 = 0,
                 command_list = None, dt = 1.):
        if command_list == None:
            headx_cmd = Command('headx',0,0,0)
            headz_cmd = Command('headz',0,0,0)
            vel_cmd   = Command('vel',0,0,0)
            command_list = [headx_cmd, headz_cmd, vel_cmd]
        super().__init__(command_list)
        self.x        = x0 # Position along x-axis.
        self.y        = y0 # Position along y-axis.
        self.z        = z0 # Position along y-axis.
        self.vel      = v0 # Velocity.
        self.headz    = hz0 # Heading around z-axis.
        self.headx    = hx0 # Heading around x-axis.
        self.dt       = dt  # Time unit

    def update(self):
        '''
        Updates the position of the model with respect to the commands. Returns
        the position and alters the model in consequence.

        Returns
        -------
        x, y , z : floats
            New positions along x-axis, y-axis and z-axis.
        '''
        cmd_headz  = self.commands['headz']
        cmd_headx  = self.commands['headx']
        cmd_vel    = self.commands['vel']
        velx       = self.vel * cos(radians(self.headx)) * cos(radians(self.headz))
        vely       = self.vel * cos(radians(self.headx)) * sin(radians(self.headz))
        velz       = self.vel * sin(radians(self.headx))
        ax, ay, az = 0, 0, 0
        self.x     += self.dt*velx # Model application: x = x + dt*velx
        self.y     += self.dt*vely # Model application: y = y + dt*vely
        self.z     += self.dt*velz # Model application: z = z + dt*velz

        if cmd_headz.steps > 0:           # Heading around z command update
            cmd_headz.steps -= 1          # Diminution of the number of steps
            self.headz += cmd_headz.delta # Adding a delta per step

        if cmd_headx.steps > 0:           # Heading around x command update
            cmd_headx.steps -= 1          # Diminution of the number of steps
            self.headx += cmd_headx.delta # Adding a delta per step

        if cmd_vel.steps > 0:           # Velocity command update
            cmd_vel.steps -= 1          # Diminution of the number of steps
            self.vel += cmd_vel.delta   # Adding a delta per step
            ax       = cmd_vel.delta * cos(radians(self.headx)) * cos(radians(self.headz))
            ay       = cmd_vel.delta * cos(radians(self.headx)) * sin(radians(self.headz))
            az       = cmd_vel.delta * sin(radians(self.headx))

        state = [self.x, velx, ax, self.y, vely, ay, self.z, velz, az]

        return state

    def change_headx(self, hdg_degrees, steps):
        '''
        Changes the heading command around x-axis.

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
        cmd_headx = self.commands['headx']
        cmd_headx.value = hdg_degrees
        cmd_headx.steps = steps
        cmd_headx.delta = cmd_headx.value / steps
        if abs(cmd_headx.delta) > 0:
            cmd_headx.steps = steps
        else:
            cmd_headx.steps = 0

        #! TODO: Refactor change_headx et change_headz to avoid duplicate code

    def change_headz(self, hdg_degrees, steps):
        '''
        Changes the heading command around z-axis.

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
        cmd_headz = self.commands['headz']
        cmd_headz.value = hdg_degrees
        cmd_headz.steps = steps
        cmd_headz.delta = cmd_headz.value / steps
        if abs(cmd_headz.delta) > 0:
            cmd_headz.steps = steps
        else:
            cmd_headz.steps = 0

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
    # Route generation example with a ManeuveredAircraft
    headx_cmd = Command('headx',0,0,0)
    headz_cmd = Command('headz',0,0,0)
    vel_cmd   = Command('vel',1,0,0)
    aircraft  = ManeuveredAircraft(x0 = 1000, y0 = 1000, z0=1, v0 = 0, hx0 = 0, hz0 = 0, command_list = [headx_cmd, headz_cmd, vel_cmd])
    xs, ys, zs = [], [], []
    states = []

    # Take off acceleration objective
    aircraft.change_command("vel",200, 20)
    # First phase -> Acceleration
    for i in range(10):
        states.append(aircraft.update())

    # Change in commands -> Take off
    aircraft.change_command("headx",45, 25)
    aircraft.change_command("headz",90, 25)

    # Second phase -> Take off
    for i in range(30):
        states.append(aircraft.update())

    # Change in commands -> Steady state
    aircraft.change_command("headx",-45, 25)
    aircraft.change_command("headz",180, 25)

    # Third phase -> Steady state
    for i in range(60):
        states.append(aircraft.update())


    states = np.array(states)
    xs = states[:,0]
    ys = states[:,1]
    zs = states[:,2]
    position_data = np.array(list(zip(xs,ys,zs)))

    # Route plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='plot test',color='k',linestyle='dashed')
    ax.scatter(xs, ys, zs,color='b',marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
