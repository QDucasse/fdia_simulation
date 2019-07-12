# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 09:10:06 2019

@author: qde
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats     import chi2
from math            import sqrt
from filterpy.common import kinematic_kf
from numpy.random    import randn
from pprint          import pprint
from fdia_simulation.helpers.plotting               import plot_measurements
from fdia_simulation.fault_detectors.fault_detector import FaultDetector

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
        # sigma = kf.R[0,0]
        # threshold = 3*sigma
        threshold = chi2.ppf(1-error_rate,kf.dim_z)
        test_quantity = sqrt(kf.y.T@kf.y)
        self.reviewed_values.append(test_quantity)
        if test_quantity <= threshold:
            self.comparison_results.append("Success")
            return "Success"
        else:
            self.comparison_results.append("Failure")
            return "Failure"



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

    # Detector instanciation
    euc_detector = EuclidianDetector()

    for z in zs:
            kinematic_test_kf.predict()
            euc_detector.review_measurement(z,kinematic_test_kf)
            kinematic_test_kf.update(z)

    print('==================EUCLIDIAN DETECTOR====================')
    zipped_euc = euc_detector.zipped_review()
    pprint(zipped_euc)

    plt.figure()
    plot_measurements(zs,alpha = 0.5)
    plt.plot(pos,'b--')
    plt.show()
