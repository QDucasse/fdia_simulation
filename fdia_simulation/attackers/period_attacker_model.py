# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:07:21 2019

@author: qde
"""

import warnings
import numpy as np
from fdia_simulation.models    import LabeledMeasurement
from fdia_simulation.attackers import Attacker,BruteForceAttacker,DriftAttacker

class PeriodAttacker(Attacker):
    '''
    Implements a basic attacker model.
    Parameters
    ----------
    filter: ExtendedKalmanFilter
        Filter of the attacked system.

    gamma: int numpy array
        Attack matrix: Identity matrix with dimension n = 3 (r,theta,phi).

    mag_vector: float numpy array
        Attack magnitude vector. The attack will consist of adding the quantity
        Gamma@mag_vector to the actual measurements.

    t0: float
        Time of beginning of the attack.

    time: int
        Duration of the attack (number of update steps)

    radar: Radar object
        Attacked radar itself.

    Attributes
    ----------
    Same as parameters +
    dim_z: int
        Dimension of the measurements.

    current_time: int
        Progression of the attack (from t0 to time)
    '''
    def __init__(self, filter, t0, time, radar, radar_pos,
                 gamma = None, mag_vector = None):
        # Store the filter and its dimension
        self.filter = filter
        self.dim_z  = filter.dim_z
        self.radar  = radar
        # The tag corresponds to the 'position' of the radar
        self.radar_tag = radar_pos
        # The position will always be the first as measurements are coming one by one
        self.radar_pos = 0
        # print('dim_z = {0}'.format(dim_z))

        # If gamma is not specified but the attacked radar position (in the
        # measurement matrix) is
        if gamma is None:
            gamma = np.eye(3)

        # If the magnitude vector is not specified but the attacked radar position
        # (in the measurement matrix) is
        if (mag_vector is None) and not(radar_pos is None):
            mag_vector = np.array([[1,1,1]]).T

        # The attack matrix should be a squared matrix with n = dim_z
        if np.shape(gamma) != (3,3):
            raise ValueError('Gamma should be a square matrix with n=3')

        if np.shape(mag_vector) != (3,1):
            msg = """The magnitude vector should have the following dimensions:
                     (3,1)"""
            raise ValueError(msg)


        self.gamma       = gamma
        self.mag_vector  = mag_vector

        # If the attackers input is null, the attack will have no effect
        if np.array_equal(self.mag_vector,np.zeros((3,1))):
            msg = """With the current attack matrix (gamma) and magnitude vector,
                     your attack will have no effect"""
            warnings.warn(msg,Warning)

        # Time attributes
        self.t0           = t0
        self.time         = time
        self.current_time = 0

    def listen_measurement(self,measurement):
        '''
        Monitors the duration (beginning and end) of the attack.
        Parameters
        ----------
        measurement: LabeledMeasurement object
            Measurement
        '''
        tag   = measurement.tag
        time  = measurement.time
        value = measurement.value
        beginning_reached = self.t0 <= self.current_time
        end_reached       = (self.current_time - self.t0) >= self.time
        in_attack         = beginning_reached and not(end_reached)
        if in_attack and (tag == self.radar_tag):
            value = self.attack_measurements(value)
        self.current_time += 1
        modified_measurement = LabeledMeasurement(time = time, tag = tag, value = value)
        return modified_measurement
