# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:54:32 2019

@author: qde
"""
from numpy.random import randn

class NoisySensor(object):
    '''Implements a noisy sensor.
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
        jitter = randn()*self.std
        return (val + jitter)

    def gen_sensor_data(self,val_list):
        '''
        Generates data of a noisy sensor with a given perioduency.
        Parameters
        ---------
        val_list: float iterable
            List of measured values.

        t: float
            Total time of the measurements.

        time_std: float
            Standard deviation of the perioduency of the measurements.

        Returns
        -------
        sensor_data: tuple list
            List of tuple composed of (time of the measurement, measurement)
        '''
        return [self.sense(val) for val in val_list]
