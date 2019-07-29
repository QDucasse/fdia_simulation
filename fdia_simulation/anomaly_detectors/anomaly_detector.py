# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:06:32 2019

@author: qde
"""
import numpy as np
from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    r'''Abstract class defining the basic function of outlier detectors.
    Attributes
    ----------
    reviewed_values: float list
        Measurements treated by the fault detector.

    comparison_results: string list
        Results of the measurements, list composed of "Success" and "Failure"
    '''
    def __init__(self):
        super().__init__()
        self.reviewed_values    = []
        self.comparison_results = []

    def zipped_review(self):
        '''
        Return a zipped list of the tested quantities and the results.
        Returns
        -------
        zipped_review: tuple iterable
            List composed of tuples (quantity tested, result)
        '''
        return  list(zip(self.reviewed_values,self.comparison_results))

    @abstractmethod
    def review_measurement(self,data,kf,error_rate):
        '''
        Abstract method that needs to be overloaded by the subclasses.
        '''
        pass
