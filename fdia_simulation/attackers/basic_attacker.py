# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:18:23 2019

@author: qde
"""

import numpy as np
from fdia_simulation.attackers import Attacker


class BasicAttacker(Attacker):
    '''
    Implements a basic attacker model.
    Parameters
    ----------
    filter: ExtendedKalmanFilter
        Filter of the attacked system.

    gamma: int numpy array
        Attack matrix: Diagonal square matrix with n = measurement (z) dimension.
        Terms on the diagonal are either equal to 0 (no attack on this parameter)
        or 1 (attack on the parameter).
        Example: 2 radars
         np.array([[0, 0, 0, 0, 0, 0],  <--- No attack on r1
                   [0, 0, 0, 0, 0, 0],  <--- No attack on theta1
                   [0, 0, 0, 0, 0, 0],  <--- No attack on phi1
                   [0, 0, 0, 1, 0, 0],  <--- Attack on r2    )
                   [0, 0, 0, 0, 1, 0],  <--- Attack on theta2 } Attack on radar2
                   [0, 0, 0, 0, 0, 1]]) <--- Attack on phi2  )

    mag_vector: float numpy array
        Attack magnitude vector. The attack will consist of adding the quantity
        Gamma@mag_vector to the actual measurements.

    t0: float
        Time of beginning of the attack.

    time: int
        Duration of the attack (number of update steps)

    Attributes
    ----------
    Same as parameters.
    '''
    def __init__(self, filter, gamma, mag_vector, t0, time):
        self.filter = filter

        dim_z = filter.dim_z
        print('dim_z = {0}'.format(dim_z))
        # The attack matrix should be a squared matrix with n = dim_z
        if np.shape(gamma) != (dim_z,dim_z):
            raise ValueError('Gamma should be a square matrix with n=dim_z')
        else:
            self.gamma = gamma

        self.mag_vector = mag_vector
        self.t0   = t0
        self.time = time

    def attack_measurements(self, z):
        return z + self.gamma@self.mag_vector
