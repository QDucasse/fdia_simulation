
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:47:18 2019

@author: qde
"""

import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve_discrete_are,lstsq
from filterpy.common import pretty_str
from filterpy.kalman import KalmanFilter
from numpy.random import randn
from fdia_simulation.attackers.attacker import Attacker, UnstableData


class YangAttacker(Attacker):
    def __init__(self):
        super().__init__()

    def compute_attack_sequence(self):
        r'''Creates the attack sequence (aka the falsified measurements passed to the filter)
        '''
        pass

    def change_measurements(self):
        r'''Alters the measurement with the attack sequence
        '''
        pass
