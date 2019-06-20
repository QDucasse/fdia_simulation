# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:06:32 2019

@author: qde
"""

from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.stats import chi2
from math import sqrt
from filterpy.common import kinematic_kf
from numpy.random import randn
from pprint import pprint
from fdia_simulation.helpers.plotting import plot_measurements


class FaultDetector(ABC):
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

    @abstractmethod
    def review_measurement(self,data,kf,error_rate):
        r'''Abstract method that needs to be overloaded by the subclasses.
        '''
        pass


class ChiSquareDetector(FaultDetector):
    r'''Fault detection based on the Chi-square statistic tests
    '''
    def __init__(self):
        super().__init__()

    def review_measurement(self,new_measurement,kf,error_rate = 0.05):
        '''
        Tests the input data and detects faulty measurements using Chi-square approach.
        Parameters
        ----------
        new_measurement: float
            New measurement given by the sensors and that needs to be verified.

        kf: KalmanFilter object
            State estimator of our model (Kalman filter here).

        error_rate: float
            Error rate within which errors are detected. For example, for an
            error rate of 0.05, we are 95% sure that the measurements obtained
            through the output of the fault detector will be correct.
        '''
        dim_z = np.shape(kf.R)[0]

        #! TODO: Raise error if wrong dimension

        # Simulated update sequence with no influence on the real filter kf
        R = kf.R
        H = kf.H
        P = kf.P
        x = kf.x
        y = new_measurement - np.dot(H, x)
        PHT = np.dot(P, H.T)
        S = np.dot(H, PHT) + R
        K = np.dot(PHT, kf.inv(S))
        x = x + np.dot(K, y)

        # Threshold calculated by reversing the chi-square table for 0.95 (by default)
        test_quantity = y.T * kf.inv(S) * y
        threshold = chi2.ppf(1-error_rate,dim_z)
        self.reviewed_values.append(*test_quantity)

        if test_quantity <= threshold:
            self.comparison_results.append("Success")
            return "Success"
        else:
            self.comparison_results.append("Failure")
            return "Failure"


class EuclidianDetector(FaultDetector):
    r'''Fault detector based on the Euclidian distance tests
    '''
    def __init__(self):
        super().__init__()

    def review_measurement(self,new_measurement,kf,error_rate = 0.05):
        r'''Tests the input data and detects faulty measurements using Euclidian distance approach
        '''
        #! TODO: sigma verification
        #! TODO: test_quantity verification
        sigma = kf.R[0,0]
        threshold = 3*sigma
        test_quantity = sqrt((np.dot(kf.H,kf.x_prior) - new_measurement)**2)
        self.reviewed_values.append(test_quantity)
        if test_quantity <= threshold:
            self.comparison_results.append("Success")
            return "Success"
        else:
            self.comparison_results.append("Failure")
            return "Failure"


class MahalanobisDetector(FaultDetector):

    # Identique à chi square ??

    def __init__(self):
        super().__init__()

    def review_measurement(self,data,kf,error_rate = 0.05):
        pass



if __name__ == "__main__":
    # Example Kalman filter for a kinematic model
    kinematic_test_kf = kinematic_kf(dim=1,order=1,dt=1)
    x         = [0.,2.]
    zs        = [x[0]]
    pos       = [x[0]]
    noise_std = 1.

    # Noisy position measurements generation for 30 samples
    for _ in range(30):
        last_pos = x[0]
        last_vel = x[1]
        new_vel  = last_vel
        new_pos  = last_pos + last_vel
        x = [new_pos, new_vel]
        z = new_pos + (randn()*noise_std)
        zs.append(z)
        pos.append(new_pos)

    # Outlier generation
    zs[5]  += 10.
    zs[10] += 10.
    zs[15] += 10.
    zs[20] += 10.
    zs[25] += 10.

    plt.figure()
    plot_measurements(zs,alpha = 0.5)
    plt.plot(pos,'b--')
    plt.show()

    # Detector instanciation
    chiDetector = ChiSquareDetector()
    eucDetector = EuclidianDetector()

    for z in zs:
            kinematic_test_kf.predict()
            chiDetector.review_measurement(z,kinematic_test_kf)
            eucDetector.review_measurement(z,kinematic_test_kf)
            kinematic_test_kf.update(z)

    print('==================CHISQUARE DETECTOR====================')
    pprint(list(zip(chiDetector.reviewed_values,chiDetector.comparison_results)))

    print('\n\n')

    print('==================EUCLIDIAN DETECTOR====================')
    pprint(list(zip(eucDetector.reviewed_values,eucDetector.comparison_results)))
