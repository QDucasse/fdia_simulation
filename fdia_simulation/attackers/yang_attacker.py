
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:47:18 2019

@author: qde
"""

import numpy as np

class YangAttacker(object):
    def __init__(self):
        super().__init__()

    def compute_attack_sequence(self):
        '''Creates the attack sequence (aka the falsified measurements passed to the filter)
        '''
        pass

    def change_measurements(self):
        '''Alters the measurement with the attack sequence
        '''
        pass
