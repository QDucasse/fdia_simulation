# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:10:54 2019

@author: qde
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import solve_discrete_are,lstsq
from filterpy.common import pretty_str
from filterpy.kalman import KalmanFilter
from numpy.random import randn


class UnstableData(object):
    r'''Implements a model for storing unstable data.
    Parameters
    ----------
    val: float
        Unstable eigenvalue.

    vect: numpy float array
        Unstable eigenvector linked to the previous eigenvalue.

    position: int
        Position of the eigenvalue in the studied matrix.
    '''
    def __init__(self,val,vect,position):
        self.value    = val
        self.vector   = vect
        self.position = position

    def __repr__(self):
        return '\n'.join([
            'UnstableData object',
            pretty_str('value', self.value),
            pretty_str('vector', self.vector),
            pretty_str('position', self.position)])


class Attacker(ABC):
    r'''Implements common behavior to attackers. 
    '''
    def __init__(self):
        pass

    @abstractmethod
    def change_measurements(self):
        pass
