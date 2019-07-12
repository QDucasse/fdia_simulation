# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:54:32 2019

@author: qde
"""

from numpy.random import randn


class NoisySensor(object):
    r'''Implements a noisy sensor.
    Parameters
    ----------
    std_noise: float
        Standard deviation of the measurement noise.

    Notes
    -----
    A NoisySensor will not generate any data itself but rather modify existing data
    to a noisy version of itself (closer to real life measurements).
    '''
    def __init__(self, std_noise=1.):
        self.std = std_noise

    def sense(self, val):
        '''
        Simulates real sensor by adding noise to a real value.
        Parameters
        ----------
        val: float
            Real value.

        Returns
        -------
        noisy_val: float
            Real value with simulated measurement noise.
        '''
        return (val + (randn()* self.std))

    def gen_sensor_data(self,val_list):
        '''
        Generates data of a noisy sensor with a given frequency.
        Parameters
        ---------
        val_list: float iterable
            List of measured values.

        t: float
            Total time of the measurements.

        time_std: float
            Standard deviation of the frequency of the measurements.

        Returns
        -------
        sensor_data: tuple list
            List of tuple composed of (time of the measurement, measurement)
        '''
        return [self.sense(val) for val in val_list]


class NoisyFrequencySensor(NoisySensor):
    r'''Implements a noisy sensor with a given frequency of data rate.
    Parameters
    ----------
    std_noise: float
        Standard deviation of the measurement noise.

    frequency: float
        Frequency of the data sampling.

    Notes
    -----
    The format of the data outputed by the NoisyFrequencySensor is no longer the same
    as NoisySensor. The time of the sampling is now included within the data list.
    '''
    def __init__(self,std_noise = 1., frequency = 1):
        if frequency == 0:
            raise ValueError("frequency can not be equal to 0")
        super().__init__(std_noise)
        self.frequency = frequency

    def gen_sensor_data(self,val_list,t,time_std):
        '''
        Generates data of a noisy sensor with a given frequency.
        Parameters
        ---------
        val_list: float iterable
            List of measured values.

        t: float
            Total time of the measurements.

        time_std: float
            Standard deviation of the frequency of the measurements.

        Returns
        -------
        sensor_data: tuple list
            List of tuple composed of (time of the measurement, measurement)
        '''
        # In case of a null time_std
        if time_std == None:
            time_std = self.frequency / 100

        sensor_data = [] # Output of the function
        dt = 0           # Time as it will be different depending on the sensor
        for i in range(t*self.frequency):
            dt    += 1/self.frequency
            t_i   =  dt + randn() * time_std   # Add noise to the time
            val_i =  np.sense(val_list[i])     # Add noise to the values
            sensor_data.append([t_i, val_i])

        return sensor_data
