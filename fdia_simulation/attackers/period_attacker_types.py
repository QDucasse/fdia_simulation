# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:07:21 2019

@author: qde
"""

import numpy as np
from fdia_simulation.attackers import PeriodAttacker,BruteForceAttacker,DriftAttacker

class BruteForcePeriodAttacker(PeriodAttacker,BruteForceAttacker):
    def __init__(self,mag = 1e6,*args,**kwargs):
        PeriodAttacker.__init__(self,*args,**kwargs)
        self.mag_vector = self.mag_vector*mag

    def listen_measurement(self,measurement):
        return PeriodAttacker.listen_measurement(self,measurement)

    def attack_measurements(self,measurement):
        return BruteForceAttacker.attack_measurements(self,measurement)

class DriftPeriodAttacker(PeriodAttacker,DriftAttacker):
    def __init__(self, attack_drift = None, *args, **kwargs):
        if attack_drift is None:
            attack_drift = np.array([[0,0,10]]).T
        self.attack_drift = attack_drift
        PeriodAttacker.__init__(self,*args,**kwargs)

    def listen_measurement(self,measurement):
        return PeriodAttacker.listen_measurement(self,measurement)

    def attack_measurements(self,measurement):
        return DriftAttacker.attack_measurements(self,measurement)
