# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 13:47:22 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models  import ManeuveredAirplane
from fdia_simulation.helpers import plot_track

class Track(object):
    r'''Implements an airplane trajectory following several modes.
    Parameters
    ----------
    airplane: ManeuveredAirplane
        The model airplane of the trajectory.
    '''
    DT_TRACK = 0.01

    def __init__(self,airplane = None, dt = None ):
        if dt is None:
            dt = self.DT_TRACK

        if airplane is None:
            airplane = ManeuveredAirplane(dt = dt)
        self.airplane = airplane

    def initial_position(self,states):
        '''
        Returns the initial position of a given array of states.
        Parameters
        ----------
        states: float numpy array
            States of the system. dim = (9,time_steps)

        Returns
        -------
        x0,y0,z0: floats
            Initial position of the observed system.
        '''
        return states[0,0], states[3,0], states[6,0]

    def gen_cruise(self,x0 = 100,y0 = 100,z0 = 8000,t = 50,vel = 250,ax='y'):
        '''
        Generates a data set for an airplane flying in steady mode along either
        x or y axis.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity during maneuver.

        ax: string
            Axis along which the airplane will move.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.vel = vel
        if ax == 'y':
            self.airplane.headz = 0

        elif ax == 'x':
            self.airplane.headz = 90

        else:
            raise ValueError('Axis must be either x or y')

        t = int(1/self.airplane.dt * t) # Consideration of the time unit

        states = []
        for _ in range(t):
            states.append(self.airplane.update())

        return np.array(states)

    def gen_weave(self, x0 = 100,y0 = 100,z0 = 8000,t = 50,vel = 250,ang = 75):
        '''
        Generates a data set for an airplane doing a weave maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity during the maneuver.

        ang: int
            Angle of the maneuver and counter maneuver.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = -ang
        self.airplane.vel = vel
        t = int(1/self.airplane.dt * t) # Consideration of the time unit
        nb_steps = t//5 # Five stages maneuver each consisting of nb_steps steps
        states = []
        # First stage: Straight line
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Second stage: Constant turn of ang degrees
        self.airplane.change_command("headz", ang, nb_steps)
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Third stage: Straight line
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Fourth stage: Constant turn of -ang degrees
        self.airplane.change_command("headz", -ang, nb_steps)
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Fifth stage: Straight line
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        return np.array(states)


    def gen_acc(self, x0 = 100,y0 = 100,z0 = 0,
                t = 20,vel = 0, end_vel = 250,
                t_acc = 5):
        '''
        Generates a data set for an airplane doing an acceleration maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver.

        end_vel: float
            Velocity at the end of the acceleration

        t_acc: int
            Duration of the acceleration

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.

        Notes
        -----
        The default values are made so that once the 25 first steps of constant
        velocity have been processed, a phase of 10 steps launches a 10g acceleration.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.vel = vel
        t = int(1/self.airplane.dt * t)         # Consideration of the time unit
        t_acc = int(1/self.airplane.dt * t_acc) # Consideration of the time unit
        nb_steps = (t - t_acc)//2
        states = []
        # First phase: Constant velocity for nb_steps steps
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Second phase: Acceleration for t_acc steps
        self.airplane.change_command("vel", end_vel, t_acc)
        for _ in range(t):
            states.append(self.airplane.update())

        # Third phase: Constant velocity for nb_steps steps
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        return np.array(states)

    def gen_dive(self, x0 = 100,y0 = 100,z0 = 4000, t = 50,vel = 30, ang = 70):
        '''
        Generates a data set for an airplane doing a dive (descent) maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity during maneuver.

        ang: int
            Angle of the descent.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.vel = vel
        states = []
        t = int(1/self.airplane.dt * t) # Consideration of the time unit
        nb_steps = t//3
        # First phase: Steady mode
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Second phase: Steady mode
        self.airplane.change_command("headx", -ang, 5)
        self.airplane.change_command("vel", 170, 5)
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        # Third phase: Recovery
        self.airplane.change_command("headx", ang, 5)
        self.airplane.change_command("vel", 100, 5)
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        return np.array(states)


    def gen_turn1(self, x0 = 100,y0 = 100,z0 = 8000, t = 50,vel = 180,
                  ang = 360, t_turn = 50):
        '''
        Generates a data set for an airplane doing a turn maneuver. By default,
        the airplane will do a 1g-turn.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver.

        ang: int
            Angle of the descent.

        t_turn: float
            Duration of the turning phase.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.vel = vel
        self.airplane.headz = 50
        states = []
        t = int(1/self.airplane.dt * t)           # Consideration of the time unit
        t_turn = int(1/self.airplane.dt * t_turn) # Consideration of the time unit
        nb_steps = t//2
        # First phase: Steady mode.
        for _ in range(nb_steps):
            states.append(self.airplane.update())
        # Second phase: 1g turn.
        self.airplane.change_command("headz", ang, t_turn)
        for _ in range(t_turn):
            states.append(self.airplane.update())

        # Third phase: Constant velocity.
        for _ in range(nb_steps):
            states.append(self.airplane.update())

        return np.array(states)


    def gen_turn5(self, x0 = 100,y0 = 100,z0 = 8000, t = 50,vel = 300,
                  ang = 360, t_turn = 15):
        '''
        Generates a data set for an airplane doing a turn maneuver. By default,
        the airplane will do a 1g-turn.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver.

        ang: int
            Angle of the descent.

        t_turn: float
            Duration of the turning phase.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        return self.gen_turn1(vel = vel, ang = ang, t_turn = t_turn)

    def gen_offensive(self, x0 = 100,y0 = 100,z0 = 8000, t = 50,vel = 150):
        '''
        Generates a data set for an airplane doing an offensive maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = 50
        self.airplane.vel = vel
        t = int(1/self.airplane.dt * t) # Consideration of the time unit
        nb_steps = t//12
        states = []

        # Constant velocity (3)
        for _ in range(3*nb_steps):
            states.append(self.airplane.update())

        # Constant turn (2) = 70° gauche
        self.airplane.change_command("headz", 50, 2*nb_steps)
        for _ in range(2*nb_steps):
            states.append(self.airplane.update())

        # Constant velocity (2)
        for _ in range(2*nb_steps):
            states.append(self.airplane.update())
        # Constant turn (3) = 160/170°
        self.airplane.change_command("headz", -140, 3*nb_steps)
        for _ in range(3*nb_steps):
            states.append(self.airplane.update())

        # Thrust acceleration (2)
        self.airplane.change_command("vel", 300, 2*nb_steps)
        for _ in range(2*nb_steps):
            states.append(self.airplane.update())

        return np.array(states)

    def gen_defensive(self, x0 = 100,y0 = 100,z0 = 8000, vel = 150, t = 100):
        '''
        Generates a data set for an airplane doing a defensive maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = 50
        self.airplane.vel = vel
        t = int(1/self.airplane.dt * t) # Consideration of the time unit
        nb_steps = t//12
        states = []

        # Constant velocity (2sec)
        for _ in range(2*nb_steps):
            states.append(self.airplane.update())
        # Constant turn (8sec) = 360°
        self.airplane.change_command("headz",360,8*nb_steps)
        for _ in range(8*nb_steps):
            states.append(self.airplane.update())
        # Thrust acceleration (2sec)
        self.airplane.change_command("vel",300,2*nb_steps)
        for _ in range(2*nb_steps):
            states.append(self.airplane.update())

        return np.array(states)

    def gen_disengagement(self, x0 = 100,y0 = 100,z0 = 8000, vel = 150, t = 100):
        '''
        Generates a data set for an airplane doing a defensive maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        t: float
            Duration of the model.

        vel: float
            Velocity at the beginning of the maneuver

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = 50
        self.airplane.vel = vel
        t = int(1/self.airplane.dt * t) # Consideration of the time unit
        nb_steps = t//12
        states = []

        # Constant velocity (3sec)
        for _ in range(3*nb_steps):
            states.append(self.airplane.update())

        # Constant turn (6sec) = 170°
        self.airplane.change_command("headz", -170,6*nb_steps)
        for _ in range(3*nb_steps):
            states.append(self.airplane.update())
        # Thrust acceleration (2sec)
        self.airplane.change_command("vel",300,3*nb_steps)
        for _ in range(3*nb_steps):
            states.append(self.airplane.update())

        return np.array(states)


    def gen_takeoff(self, x0 = 100,y0 = 100,z0 = 0):
        '''
        Generates a data set for an airplane doing a takeoff maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = 50
        self.airplane.vel = 0
        states = []
        self.airplane.change_command("vel", 80, int(1/self.airplane.dt*20))
        for _ in range(int(1/self.airplane.dt*20)):
            states.append(self.airplane.update())

        self.airplane.change_command("vel", 200, int(1/self.airplane.dt*100))
        self.airplane.change_command("headx", 30, int(1/self.airplane.dt*5))
        for _ in range(int(1/self.airplane.dt*5)):
            states.append(self.airplane.update())

        self.airplane.change_command("headz", 70, int(1/self.airplane.dt*7))
        for _ in range(int(1/self.airplane.dt*20)):
            states.append(self.airplane.update())

        self.airplane.change_command("headz", 70, int(1/self.airplane.dt*7))
        for _ in range(int(1/self.airplane.dt*30)):
            states.append(self.airplane.update())

        self.airplane.change_command("headz", 70, int(1/self.airplane.dt*7))
        self.airplane.change_command("headx", -30, int(1/self.airplane.dt*7))
        for _ in range(int(1/self.airplane.dt*30)):
            states.append(self.airplane.update())

        return np.array(states)

    def gen_landing(self, x0 = 1200, y0 = 1200, z0 = 8000):
        '''
        Generates a data set for an airplane doing a takeoff maneuver.
        Parameters
        ----------
        x0, y0, z0: floats
            Initial position of the airplane in steady mode.

        Returns
        -------
        xs, ys, zs: float iterables
            Positions generated along the three axis.
        '''
        self.airplane.x   = x0
        self.airplane.y   = y0
        self.airplane.z   = z0
        self.airplane.headz = 50
        self.airplane.vel = 300
        states = []

        for _ in range(int(1/self.airplane.dt*20)):
            states.append(self.airplane.update())

        self.airplane.change_command("headz", 70, int(1/self.airplane.dt*7))
        self.airplane.change_command("headx", -30, int(1/self.airplane.dt*7))
        for _ in range(int(1/self.airplane.dt*30)):
            states.append(self.airplane.update())

        self.airplane.change_command("headz", 70, int(1/self.airplane.dt*7))
        for _ in range(int(1/self.airplane.dt*25)):
            states.append(self.airplane.update())

        self.airplane.change_command("headx", 30, int(1/self.airplane.dt*7))
        self.airplane.change_command("vel",0, int(1/self.airplane.dt*20))
        for _ in range(int(1/self.airplane.dt*20)):
            states.append(self.airplane.update())

        return np.array(states)


def output_positions(states):
        return states[:,0],states[:,3],states[:,6]


if __name__ == "__main__":
    test_track = Track()
    print('##== Figure 1: Cruise mode =========##')
    xs_cruise, ys_cruise, zs_cruise = output_positions(test_track.gen_cruise())
    plot_track(xs_cruise, ys_cruise, zs_cruise, 1, 'Cruise trajectory, dt = 1.')

    test_track = Track()
    print('##== Figure 2: Weave mode ==========##')
    xs_weave, ys_weave, zs_weave = output_positions(test_track.gen_weave())
    plot_track(xs_weave, ys_weave, zs_weave, 2, 'Weave trajectory')

    test_track = Track()
    print('##== Figure 3: Acceleration mode ===##')
    xs_acc, ys_acc, zs_acc = output_positions(test_track.gen_acc())
    plot_track(xs_acc, ys_acc, zs_acc, 3, 'Acceleration trajectory')

    test_track = Track()
    print('##== Figure 4: Dive mode ===========##')
    xs_dive, ys_dive, zs_dive = output_positions(test_track.gen_dive())
    plot_track(xs_dive, ys_dive, zs_dive, 4, 'Dive trajectory')

    test_track = Track()
    print('##== Figure 5: 1g turn mode ========##')
    xs_turn1, ys_turn1, zs_turn1 = output_positions(test_track.gen_turn1())
    plot_track(xs_turn1, ys_turn1, zs_turn1, 5, '1g Turn trajectory')

    test_track = Track()
    print('##== Figure 6: 5g turn mode ========##')
    xs_turn5, ys_turn5, zs_turn5 = output_positions(test_track.gen_turn5())
    plot_track(xs_turn5, ys_turn5, zs_turn5, 6, '5g Turn trajectory')

    test_track = Track()
    print('##== Figure 7: Offensive mode ======##')
    xs_off, ys_off, zs_off = output_positions(test_track.gen_offensive())
    plot_track(xs_off, ys_off, zs_off, 7, 'Offensive trajectory')

    test_track = Track()
    print('##== Figure 8: Defensive mode ======##')
    xs_def, ys_def, zs_def = output_positions(test_track.gen_defensive())
    plot_track(xs_def, ys_def, zs_def, 8, 'Defensive trajectory')

    test_track = Track()
    print('##== Figure 9: Defensive mode ======##')
    xs_dis, ys_dis, zs_dis = output_positions(test_track.gen_disengagement())
    plot_track(xs_dis, ys_dis, zs_dis, 9, 'Disengagement trajectory')

    test_track = Track(dt = 0.4)
    print('##== Figure 10: Takeoff mode =======##')
    xs_toff, ys_toff, zs_toff = output_positions(test_track.gen_takeoff())
    plot_track(xs_toff, ys_toff, zs_toff, 10, 'Takeoff trajectory')

    test_track = Track(dt = 0.4)
    print('##== Figure 11: Landing mode =======##')
    xs_land, ys_land, zs_land = output_positions(test_track.gen_landing())
    plot_track(xs_land, ys_land, zs_land, 11, 'Landing trajectory')

    plt.show()
