# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:10:02 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from math                              import sqrt
from pprint                            import pprint
from numpy.linalg                      import inv
from numpy.random                      import randn
from filterpy.common                   import kinematic_kf
from fdia_simulation.helpers           import plot_measurements
from fdia_simulation.filters           import RadarFilterModel, MultiplePeriodRadarsFilterModel
from fdia_simulation.anomaly_detectors import AnomalyDetector


class MahalanobisDetector(AnomalyDetector):
    '''
    Fault detection based on the Mahalanobis distance put against the
    precomputed threshold based on chi-squared distribution.
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
        # dim_z = filter.dim_z
        # # The measurement matrix has to be computed if the filter is Eself.filter like
        if isinstance(filter,MultiplePeriodRadarsFilterModel):
            tag = measurement[0]
            z   = measurement[1]
            H = filter.HJacob(filter.x,tag)
            measurement = z
        elif isinstance(filter,RadarFilterModel):
            H = filter.HJacob(filter.x)
        else:
            H = filter.H

        # Simulated update sequence with no influence on the real filter self.filter
        y   = measurement - H@filter.x # Residual:              z - Hx
        PHT = filter.P@H.T             # Intermediate variable: P*H*T
        S   = H@PHT + filter.R         # Innovation covariance: H*P*HT + R
        # K   = PHT@inv(S)           # Kalman gain:           P*HT*inv(S)
        # x   = self.filter.x + K@y           # New state:             x + Ky

        test_quantity = sqrt(y.T @ inv(S) @ y)

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
    mah_detector = MahalanobisDetector()
    print(mah_detector.threshold)

    for z in zs:
            kinematic_test_kf.predict()
            if mah_detector.review_measurement(z,kinematic_test_kf):
                kinematic_test_kf.update(z)
            else:
                kinematic_test_kf.update(None)

    print('==================Mahalanobis DETECTOR====================')
    zipped_mah = mah_detector.zipped_review()
    pprint(zipped_mah)

    plt.figure()
    plot_measurements(zs,alpha = 0.5)
    plt.plot(pos,'b--')
    plt.show()
