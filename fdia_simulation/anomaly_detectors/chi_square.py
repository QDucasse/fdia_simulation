# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:10:02 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from pprint                          import pprint
from scipy.stats                     import chi2
from numpy.linalg                    import inv
from numpy.random                    import randn
from filterpy.common                 import kinematic_kf
from fdia_simulation.helpers         import plot_measurements
from fdia_simulation.anomaly_detectors import AnomalyDetector


class ChiSquareDetector(AnomalyDetector):
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

        Returns
        -------
        result: boolean
            True or False.

        Notes
        -----
        The returned boolean is also added to the instance variable
        comparison_results.
        '''
        dim_z = np.shape(kf.R)[0]

        #! TODO: Raise error if wrong dimension
        from fdia_simulation.filters.radar_filter_model import RadarModel
        if isinstance(kf,RadarModel):
            H = kf.HJacob(kf.x)
        else:
            H = kf.H

        # Simulated update sequence with no influence on the real filter kf
        y   = kf.residual_of(new_measurement) # Residual:              z - Hx
        PHT = kf.P@H.T                        # Intermediate variable: P*H*T
        S   = H@PHT + kf.R                    # Innovation covariance: H*P*HT + R
        K   = PHT@inv(S)                      # Kalman gain:           P*HT*inv(S)
        x   = kf.x + K@y                   # New state:             x + Ky

        # Threshold calculated by reversing the chi-square table for 0.95 (by default)
        test_quantity = y.T @ inv(S) @ y
        threshold = chi2.ppf(1-error_rate,dim_z)
        self.reviewed_values.append(test_quantity)

        if test_quantity <= threshold:
            self.comparison_results.append(True)
            return True
        else:
            self.comparison_results.append(False)
            return False


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
    chi_detector = ChiSquareDetector()

    for z in zs:
            kinematic_test_kf.predict()
            if chi_detector.review_measurement(z,kinematic_test_kf):
                kinematic_test_kf.update(z)
            else:
                kinematic_test_kf.update(None)

    print('==================CHISQUARE DETECTOR====================')
    zipped_chi = chi_detector.zipped_review()
    pprint(zipped_chi)

    plt.figure()
    plot_measurements(zs,alpha = 0.5)
    plt.plot(pos,'b--')
    plt.show()
