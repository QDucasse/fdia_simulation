# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:06:32 2019

@author: qde
"""
import numpy as np
from abc         import ABC, abstractmethod
from scipy.stats import chi2

class AnomalyDetector(ABC):
    '''Abstract class defining the use of anomaly detectors. Designed to be a
    part of a filter.
    Attributes
    ----------
    reviewed_values: float list
        Measurements treated by the fault detector.

    comparison_results: string list
        Results of the measurements, list composed of "Success" and "Failure"
    '''
    def __init__(self,error_rate = 0.05):
        super().__init__()
        self.reviewed_values    = []
        self.comparison_results = []
        self.error_rate = error_rate
        self.threshold  = None

    def zipped_review(self):
        '''
        Return a zipped list of the tested quantities and the results.
        Returns
        -------
        zipped_review: tuple iterable
            List composed of tuples (quantity tested, result)
        '''
        return  list(zip(self.reviewed_values,self.comparison_results))

    def review_measurement(self,measurement,filter):
        '''
        Computes the tested quantity from the measurement and puts it against
        the threshold of the detector.
        Parameters
        ----------
        measurement: float numpy array
            The measurement coming from a radar and needing to be tested.

        Returns
        -------
        res: boolean
            Acceptation or refusal of the incoming measurement.
        '''
        if self.threshold is None:
            self.compute_threshold(dim_z = filter.dim_z)
        test_quantity = self.compute_test_quantity(measurement,filter)
        res = self.compare_test_quantity(test_quantity)
        return res

    def compute_threshold(self,dim_z,error_rate = None):
        '''
        Computes the treshold that will be used as a comparison criteria by
        "reversing" the chi-squared distribution with a given error_rate.
        We are looking for the threshold in the equation:
        error_rate = P(test_quantity > threshold)
        Parameters
        ----------
        dim_z: int
            Size of the measurement vector.

        error_rate: float
            Probability of a measurement to be an error. Default value of 5%.
        '''
        if error_rate is None:
            error_rate = self.error_rate
        threshold = chi2.ppf(1-error_rate,dim_z)
        self.threshold = threshold

    def compare_test_quantity(self,test_quantity):
        '''
        Compares the tested quantity (depending of the type of detector, please
        see compute_test_quantity()) to the threshold. Returns a boolean and
        stores it in the comparison_results variable. The reviewed quantity is
        stored in the reviewed_values variable.
        Parameters
        ----------
        test_quantity: float
            The quantity to be compared to the threshold. Depends on the type
            of detector.

        Returns
        -------
        res: boolean
            Result of the comparison, determines if the measurement is considered
            accepted (True) or rejected (False).
        '''
        self.reviewed_values.append(test_quantity)
        if test_quantity <= self.threshold:
            self.comparison_results.append(True)
            return True
        else:
            self.comparison_results.append(False)
            return False

    @abstractmethod
    def compute_test_quantity(self,measurement,filter):
        '''
        Computes the float that will be put against the threshold to determine
        wether or not the measurement is correct. Method depends on the type of
        detector. Must be overloaded by subclasses.
        Parameters
        ----------
        measurement: float numpy array
            Measurement coming from a radar and needed to be tested.
        '''
        pass
