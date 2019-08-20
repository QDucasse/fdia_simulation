# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 09:10:06 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from math                            import sqrt
from pprint                          import pprint
from scipy.stats                     import chi2
from numpy.random                    import randn
from filterpy.common                 import kinematic_kf
from fdia_simulation.helpers         import plot_measurements
from fdia_simulation.anomaly_detectors import AnomalyDetector

class EuclidianDetector(AnomalyDetector):
    '''
    Fault detector based on the Euclidian distance tests
    '''

    def compute_test_quantity(self,measurement,filter):
        '''
        Computes the float that will be put against the threshold to determine
        wether or not the measurement is correct.
        Parameters
        ----------
        measurement: float numpy array
            Measurement coming from a radar and needed to be tested.

        self.filter: KalmanFilter object
            The Kalman filter the detector is attached to.
        '''
        test_quantity = sqrt(filter.y.T@filter.y)

        return test_quantity

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
