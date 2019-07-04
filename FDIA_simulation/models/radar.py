# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:46 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin,sqrt,pi,atan2
from fdia_simulation.models.moving_target import MovingTarget, Command
from fdia_simulation.models.maneuvered_aircraft import ManeuveredAircraft
from fdia_simulation.models.sensors import NoisySensor

class Radar(object):
    r'''Implements a simulated radar.
    The radar will output a data set corresponding to typical radar values.
    Attributes
    ----------
    x, y, z: floats
        Radar position along x, y and z-axis.

    r_noise_std: float
        Standard deviation on the measurement of r. Default value of 1.

    theta_noise_std: float
        Standard deviation on the measurement of theta. Default value of 0.1

    phi_noise_std: float
        Standard deviation on the measurement of phi. Default value of 0.1

    Parameters
    ----------
    Identical to Attributes
    '''

    def __init__(self, x, y, z=0, r_noise_std = 1., theta_noise_std = 0.001, phi_noise_std = 0.001):
        self.x                = x
        self.y                = y
        self.z                = z
        self.r_noise_std      = r_noise_std
        self.theta_noise_std  = theta_noise_std
        self.phi_noise_std    = phi_noise_std
        self.R = np.array([[r_noise_std,0              ,0            ],
                           [0          ,theta_noise_std,0            ],
                           [0          ,0              ,phi_noise_std]])

    def get_position(self):
        '''
        Position accessor.
        Returns
        -------
        position: float iterable
            [x,y,z] of the radar.
        '''
        return [self.x,self.y,self.z]

    def gen_data(self,position_data):
        '''
        Generates simulated received data for a radar.
        Parameters
        ----------
        position_data: float list numpy array
            List of positions in the form of lists [x_k, y_k, z_k].
            Corresponding to:
            x_k: float
                Position along x-axis.

            y_k: float
                Position along y-axis.

            z_k: float
                Position along z-axis.

        Returns
        -------
        rs, thetas, phis: float iterables
            Distances, azimuth/turn angles and elevation angles.
        '''
        xs = position_data[:,0] - self.x
        ys = position_data[:,1] - self.y
        zs = position_data[:,2] - self.z
        rs, thetas, phis = [], [], []

        for k in range(len(position_data)):
            # Computation of the supposed distance of the aircraft
            r_k     = sqrt(xs[k]**2 + ys[k]**2 + zs[k]**2)

            # Computation of the supposed turning angle of the aircraft
            theta_k = atan2(ys[k],xs[k])

            # Computation of the supposed elevation angle of the aircraft
            phi_k   = atan2(zs[k], sqrt(xs[k]**2 + ys[k]**2))

            rs.append(r_k)
            thetas.append(theta_k)
            phis.append(phi_k)

        return rs, thetas, phis

    def sense(self, rs, thetas, phis):
        '''
        Simulates real sensors by adding noise to the predicted simulated values.
        Parameters
        ----------
        rs, thetas, phis: float iterable
            Distances, azimuth/turn angles and elevation angles.

        Returns
        -------
        noisy_rs, noisy_thetas, noisy_phis: float iterable
            Distances, azimuth/turn angles and elevation angles with added white noise.
        '''

        nsr     = NoisySensor(self.r_noise_std)
        nstheta = NoisySensor(self.theta_noise_std)
        nsphi   = NoisySensor(self.phi_noise_std)

        noisy_rs     = [nsr.sense(r) for r in rs]
        noisy_thetas = [nstheta.sense(theta) for theta in thetas]
        noisy_phis   = [nsphi.sense(phi) for phi in phis]

        return noisy_rs, noisy_thetas, noisy_phis


    def radar2cartesian(self,rs,thetas,phis):
        '''
        Transcripts the radar measured values (r, theta, phi) to cartesian
        positions (x, y, z).
        Parameters
        ----------
        rs: float iterable
            List of the rs (distance) measured by the radar.

        thetas: float iterable
            List of the thetas (azimuth/turn angle) measured by the radar.

        phis: float iterable
            List of the phis (elevation angle) measured by the radar.

        Returns
        -------
        xs: float iterable
            List of the computed positions along x-axis.

        ys: float iterable
            List of the computed positions along y-axis.

        zs: float iterable
            List of the computed positions along z-axis.
        '''
        xs,ys,zs = [],[],[]
        for k in range(len(rs)):
            x_k = rs[k] * cos(thetas[k]) * cos(phis[k]) + self.x
            y_k = rs[k] * sin(thetas[k]) * cos(phis[k]) + self.y
            z_k = rs[k] * sin(phis[k]) + self.z

            xs.append(x_k)
            ys.append(y_k)
            zs.append(z_k)

        return xs,ys,zs

if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    headx_cmd = Command('headx',0,0,0)
    headz_cmd = Command('headz',0,0,0)
    vel_cmd   = Command('vel',1,0,0)
    aircraft  = ManeuveredAircraft(x0 = 1000, y0 = 1000, z0=1, v0 = 0, hx0 = 0, hz0 = 0, command_list = [headx_cmd, headz_cmd, vel_cmd])
    xs, ys, zs = [], [], []

    # Take off acceleration objective
    aircraft.change_command("vel",200, 20)
    # First phase -> Acceleration
    for i in range(10):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Change in commands -> Take off
    aircraft.change_command("headx",45, 25)
    aircraft.change_command("headz",90, 25)

    # Second phase -> Take off
    for i in range(30):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Change in commands -> Steady state
    aircraft.change_command("headx",-45, 25)
    aircraft.change_command("headz",180, 25)

    # Third phase -> Steady state
    for i in range(60):
        x, y, z = aircraft.update()
        xs.append(x)
        ys.append(y)
        zs.append(z)

    position_data = np.array(list(zip(xs,ys,zs)))
    print("Aircraft position:\n{0}\n".format(position_data[-25:,:]))
    # ==========================================================================
    # ========================== Radars generation =============================
    radar = Radar(x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    # print("Noisy radar data:\n{0}\n".format(radar_values[-25:,:]))

    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    # print("Estimated positions:\n{0}\n".format(radar_computed_values[-25:,:]))
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure()
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real airplane position', color='k', linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad, color='b', marker='o', label='Measurements')
    ax.scatter(radar.x,radar.y,radar.z,color='r',label='Radar position')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.show()
