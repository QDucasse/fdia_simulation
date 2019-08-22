# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:46:31 2019

@author: qde
"""

from filterpy.kalman                   import IMMEstimator

class RadarIMM(IMMEstimator):
    def __init__(self,*args,**kwargs):
        IMMEstimator.__init__(self,*args,**kwargs)
        self.dim_z = self.filters[0].dim_z

    def update(self,measurement):
        if max(self.mu)>0.8:
            for filter in self.filters:
                filter.activate_detection()

        IMMEstimator.update(self,measurement)
