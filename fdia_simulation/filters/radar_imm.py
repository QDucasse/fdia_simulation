# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:46:31 2019

@author: qde
"""

from filterpy.kalman import IMMEstimator

class RadarIMM(IMMEstimator):
    '''
    Implements an Interacting Multiple Model filter for our radar.

    Note:
    -----
    This class is a simple extension of the filterpy's IMM class.
    '''
    # def __init__(self,*args,**kwargs):
    #     IMMEstimator.__init__(self,*args,**kwargs)
    #     self.dim_z = self.filters[0].dim_z

    def update(self,measurement):
        '''
        The update takes in account any anomaly detector if one of the model has
        a probability above 0.8. This feature allows the anomaly to not be taken
        in consideration when the system is doing a maneuver. Switching model is
        not breaking the estimation that way.
        Parameters
        ----------
        measurement: float numpy array
            The measurement added to the update cycle of the filter.
        '''
        if max(self.mu)>0.8:
            for filter in self.filters:
                filter.activate_detection()

        IMMEstimator.update(self,measurement)
