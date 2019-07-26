# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:07:46 2019

@author: qde
"""
import numpy as np
import matplotlib.pyplot as plt
from math                   import cos,sin,sqrt,pi,atan2
from numpy.random           import randn
from filterpy.common        import pretty_str
from fdia_simulation.models import ManeuveredAircraft, NoisySensor, Track, MovingTarget, Command

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

    DT_RADAR = 0.1

    def __init__(self, x = 0, y = 0, z = 0, dt = None,
                 r_noise_std = 1., theta_noise_std = 0.001, phi_noise_std = 0.001):

        if dt is None:
            dt = self.DT_RADAR
        self.dt   = dt
        self.x    = x
        self.y    = y
        self.z    = z
        self.step = self.dt / Track.DT_TRACK # Sampling step from the position data
        self.r_noise_std     = r_noise_std
        self.theta_noise_std = theta_noise_std
        self.phi_noise_std   = phi_noise_std
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

    def gen_radar_values(self,x,y,z):
        '''
        Computes the three parameters r, theta and phi from the given positions.
        Parameters
        ----------
        x,y,z: floats
            Position of the aircraft.

        Returns
        -------
        r,theta,phi: floats
            Radar values corresponding to the input position.
        '''
        # Importance of the radar position
        x -= self.x
        y -= self.y
        z -= self.z

        # Computation of the distance of the aircraft
        r = sqrt(x**2 + y**2 + z**2)
        # Computation of the turning angle of the aircraft
        theta = atan2(y,x)
        # Computation of the elevation angle of the aircraft
        phi = atan2(z, sqrt(x**2 + y**2))

        return r, theta, phi

    def sample_position_data(self,position_data):
        '''
        Samples the initial position data (computed with dt = 0.01) to reduce it
        to the actual data rate of the radar.
        '''
        sampled_position_data = position_data[::int(self.step)]
        return sampled_position_data

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
        rs, thetas, phis = [], [], []

        for position in position_data:
            x_k = position[0]
            y_k = position[1]
            z_k = position[2]
            # Computation of the supposed distance of the aircraft
            r_k, theta_k, phi_k = self.gen_radar_values(x_k,y_k,z_k)

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
        nsr     = NoisySensor(std_noise = self.r_noise_std)
        nstheta = NoisySensor(std_noise = self.theta_noise_std)
        nsphi   = NoisySensor(std_noise = self.phi_noise_std)

        noisy_rs     = [nsr.sense(r) for r in rs]
        noisy_thetas = [nstheta.sense(theta) for theta in thetas]
        noisy_phis   = [nsphi.sense(phi) for phi in phis]

        return noisy_rs, noisy_thetas, noisy_phis

    def gen_position_vals(self,r,theta,phi):
        '''
        Compute the position from the radar values r, theta and phi.
        Parameters
        ----------
        r,theta,phi: floats
            Radar values.

        Returns
        -------
        x,y,z: floats
            Sensed position of the aircraft extracted from the measurement given
            in input.
        '''
        x = r * cos(theta) * cos(phi) + self.x
        y = r * sin(theta) * cos(phi) + self.y
        z = r *              sin(phi) + self.z
        return x,y,z


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
        for r,theta,phi in zip(rs,thetas,phis):
            x_k,y_k,z_k = self.gen_position_vals(r,theta,phi)
            xs.append(x_k)
            ys.append(y_k)
            zs.append(z_k)

        return xs,ys,zs

    def __eq__(self,other):
        eq_dt  = (self.dt == other.dt)
        eq_pos = (
                    (self.y == other.y) and
                    (self.x == other.x) and
                    (self.z == other.z)
                 )
        eq_std = (
                    (self.r_noise_std     == other.r_noise_std)     and
                    (self.theta_noise_std == other.theta_noise_std) and
                    (self.phi_noise_std   == other.phi_noise_std)
                 )
        return all([eq_dt,eq_pos,eq_std])

class LabeledMeasurement(object):
    '''
    Measurement labeled with a tag (radar ownership) and a timestamp (date of the
    measurement).
    Parameters
    ----------
    tag: int
        Tag (position) of the radar emitting this measurement.

    time: float
        Time of the measurement, starting from the beginning of the observation.

    value: float numpy array
        Array containing [r, theta, phi], measurement of tagged radar at the
        given time.
    '''
    def __init__(self,tag,time,value):
        self.tag   = tag
        self.time  = time
        self.value = value

    '''
    Redifinition of the comparison operators using as only criteria the time of
    measurement.
    '''
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
            'LabeledMeasurement object',
            pretty_str('tag', self.tag),
            pretty_str('time', self.time),
            pretty_str('value', self.value)])

class FrequencyRadar(Radar):
    r'''
    Implements a radar with a given data rate (dt).
    Attributes
    ----------
    Radar attributes +
    dt: float
        Data rate of the radar.

    tag: int
        Radar tag (position in the radars list).

    time_std: float
        Standard deviation of the time. Default value of 0.001

    Parameters
    ----------
    Identical to attributes
    '''
    def __init__(self, x, y, z=0, dt = None,
                 r_noise_std = 1., theta_noise_std = 0.001, phi_noise_std = 0.001,
                 time_std = 0.001):

        if dt is None:
            dt = Radar.DT_RADAR
        self.time_std = time_std
        self.tag      = 0
        Radar.__init__(self,x = x, y = y, z = z, dt = dt,
                       r_noise_std = r_noise_std, theta_noise_std = theta_noise_std, phi_noise_std = phi_noise_std)


    def compute_meas_times(self, size):
        '''
        Computes the measurement times adding repeatitively dt (modified by the
        time_std).
        Parameters
        ----------
        size: int
            Size of the list of times.

        Returns
        -------
        meas_times: float list
            List of the sample times.
        '''
        t_k = 0
        meas_times = [t_k]
        for _ in range(size-1):
            t_k += self.dt + randn()*self.time_std # Adding a time jitter
            meas_times.append(t_k)
        return meas_times


    def compute_measurements(self,position_data):
        '''
        Computes the measurements of given positions.
        Parameters
        ----------
        position_data: float numpy array
            Array of positions [x,y,z].

        Returns
        -------
        measurements: LabeledMeasurement list
            List of labeled measurements with time and tag.
        '''
        rs, thetas, phis = self.gen_data(position_data)
        noisy_rs, noisy_thetas, noisy_phis = self.sense(rs, thetas, phis)
        n = len(noisy_rs)
        measurement_times = self.compute_meas_times(n)
        measurements = []

        for i in range(n):
            value = [noisy_rs[i], noisy_thetas[i], noisy_phis[i]]
            measurement = LabeledMeasurement(tag = self.tag,
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
