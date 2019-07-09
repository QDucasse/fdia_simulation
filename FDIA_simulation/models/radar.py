# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:46 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from math            import cos,sin,sqrt,pi,atan2
from numpy.random    import randn
from filterpy.common import pretty_str
from fdia_simulation.models.moving_target       import MovingTarget, Command
from fdia_simulation.models.maneuvered_aircraft import ManeuveredAircraft
from fdia_simulation.models.sensors             import NoisySensor
from fdia_simulation.models.tracks              import Track

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

class LabelizedMeasurement(object):
    def __init__(self,tag,time,value):
        self.tag   = tag
        self.time  = time
        self.value = value

    def __gt__(self,other):
        return (self.time > other.time)

    def __ge__(self,other):
        return (self.time >= other.time)

    def __le__(self,other):
        return (self.time <= other.time)

    def __lt__(self,other):
        return (self.time < other.time)

    def __repr__(self):
        return '\n'.join([
            'LabelizedMeasurement object',
            pretty_str('tag', self.tag),
            pretty_str('time', self.time),
            pretty_str('value', self.value)])

class FrequencyRadar(Radar):
    r'''
    '''
    def __init__(self, x, y, z=0, tag = 0,
                 r_noise_std = 1., theta_noise_std = 0.001, phi_noise_std = 0.001,
                 dt = 1., time_std = 0.001):

        self.dt       = dt
        self.tag      = tag
        self.time_std = time_std
        Radar.__init__(self,x = x, y = y, z = z,
                       r_noise_std = r_noise_std, theta_noise_std = theta_noise_std, phi_noise_std = phi_noise_std)

    def compute_meas_times(self, size):
        '''
        '''
        meas_times = [0]
        t_k = meas_times[0]
        for _ in range(size-1):
            t_k += self.dt + randn()*self.time_std # Adding a time jitter
            meas_times.append(t_k)
        return meas_times

    def compute_measurements(self,position_data):
        '''
        '''
        rs, thetas, phis = self.gen_data(position_data)
        noisy_rs, noisy_thetas, noisy_phis = self.sense(rs, thetas, phis)
        n = len(noisy_rs)
        measurement_times = self.compute_meas_times(n)
        measurements = []

        for i in range(n):
            value = [noisy_rs[i], noisy_thetas[i], noisy_phis[i]]
            measurement = LabelizedMeasurement(tag = self.tag,
                                               time = measurement_times[i],
                                               value = value)
            measurements.append(measurement)

        return measurements


if __name__ == "__main__":
    #================== Positions generation for the aircraft ==================
    trajectory = Track(dt = 0.1)
    xs, ys, zs = trajectory.gen_takeoff()
    position_data = np.array(list(zip(xs,ys,zs)))

    trajectory2 = Track(dt = 0.4)
    xs2, ys2, zs2 = trajectory2.gen_takeoff()
    position_data2 = np.array(list(zip(xs2,ys2,zs2)))
    # ==========================================================================
    # ========================== Radars generation =============================
    # Radar 1
    radar = FrequencyRadar(tag = 0, x=800,y=800)
    rs, thetas, phis = radar.gen_data(position_data)
    noisy_rs, noisy_thetas, noisy_phis = radar.sense(rs, thetas, phis)
    xs_from_rad, ys_from_rad, zs_from_rad = radar.radar2cartesian(noisy_rs, noisy_thetas, noisy_phis)

    radar_values = np.array(list(zip(noisy_rs, noisy_thetas, noisy_phis)))
    # print("Noisy radar data:\n{0}\n".format(radar_values[-25:,:]))

    radar_computed_values = np.array(list(zip(xs_from_rad, ys_from_rad, zs_from_rad)))
    # print("Estimated positions:\n{0}\n".format(radar_computed_values[-25:,:]))

    measurements_info = radar.compute_measurements(position_data)
    print("Measurements radar1:\n{0}\n".format(measurements_info[-25:]))

    # Radar 2
    radar2 = FrequencyRadar(tag = 1, x=1000,y=1000, r_noise_std = 5., theta_noise_std = 0.005, phi_noise_std = 0.005)
    rs2, thetas2, phis2 = radar2.gen_data(position_data2)
    noisy_rs2, noisy_thetas2, noisy_phis2 = radar2.sense(rs2, thetas2, phis2)
    xs_from_rad2, ys_from_rad2, zs_from_rad2 = radar2.radar2cartesian(noisy_rs2, noisy_thetas2, noisy_phis2)

    radar_values2 = np.array(list(zip(noisy_rs2, noisy_thetas2, noisy_phis2)))
    # print("Noisy radar data:\n{0}\n".format(radar_values[-25:,:]))

    radar_computed_values2 = np.array(list(zip(xs_from_rad2, ys_from_rad2, zs_from_rad2)))
    # print("Estimated positions:\n{0}\n".format(radar_computed_values[-25:,:]))

    measurements_info2 = radar2.compute_measurements(position_data2)
    print("Measurements radar2:\n{0}\n".format(measurements_info2[-25:]))


    measurement_infos = (measurements_info + measurements_info2).sort()
    # ==========================================================================
    # =============================== Plotting =================================
    fig = plt.figure(1)
    plt.rc('font', family='serif')
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs, label='Real airplane position', color='k', linestyle='dashed')
    ax.scatter(xs_from_rad, ys_from_rad, zs_from_rad, color='b', marker='o', label='Precision radar measurements')
    ax.scatter(xs_from_rad2, ys_from_rad2, zs_from_rad2, color='m', marker='o', label='Standard radar measurements')
    ax.scatter(radar.x,radar.y,radar.z,color='r',label='Precision radar')
    ax.scatter(radar2.x,radar2.y,radar2.z,color='r',label='Standard radar')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    plt.show()
