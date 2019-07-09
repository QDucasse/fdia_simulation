# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:43:57 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from pprint import pprint
from filterpy.kalman import KalmanFilter
from fdia_simulation.attackers.mo_attacker      import MoAttacker
from fdia_simulation.fault_detectors.chi_square import ChiSquareDetector
from fdia_simulation.fault_detectors.euclidian  import  EuclidianDetector


if __name__ == "__main__":
    #  ================== Filter and system generation ========================
    # Here we will be using a simple system following x-axis with constant
    # velocity and equiped with both position and velocity sensors.
    kf = KalmanFilter(dim_x=2,dim_z=2)
    kf.x      = [0.,2.]   # Initial values state-space is [ x xdot ]'
    kf.H      = np.array([[1.,0.],  # Observation matrix
                          [0.,1.]])
    kf.F      = np.array([[1., 1.],
                          [0., 1.]]) # State transition matrix
    kf.R      = np.eye(2)
    kf.Q      = np.eye(2)
    kf.P      = np.array([[1., 0.],
                          [0., 1.]])
    kf.B      = np.array([[0.5, 1.]]).T
    xs        = kf.x      # Initial values for the data generation
    zs        = [[kf.x[0],kf.x[1]]] # We are measuring both the position and velocity
    pos       = [kf.x[0]] # Initialization of the true position values
    vel       = [kf.x[1]] # Initialization of the true velocity values
    noise_std = 1.

    # ==========================================================================
    # ======== Noisy position measurements generation for 30 samples ===========
    sample_nb = 1000

    for _ in range(sample_nb-1):
        last_pos = xs[0]
        last_vel = xs[1]
        new_vel  = last_vel
        new_pos  = last_pos + last_vel
        xs       = [new_pos, new_vel]
        z        = [new_pos + (randn()*noise_std), new_vel + (randn()*noise_std)]

        zs.append(z)
        pos.append(new_pos)
        vel.append(new_vel)

    # ==========================================================================
    # ========================= Attacker generation ============================

    mo_attacker = MoAttacker(kf)

    # Computation of the whole attack sequence
    zas, Gamma = mo_attacker.compute_attack_sequence(attack_size = sample_nb, pos_value = 1, logs = True)

    # Adding the attack sequence to the real measurements
    zs = np.array(zs).T

    for i in range(len(zas[0])):
        zs[:,i] = zs[:,i] + Gamma@(zas[:,i])

    # ==========================================================================
    # ======================== State estimation ================================
    # Fault detector initialization
    fault_detector = ChiSquareDetector()

    xs = [] # Estimated states
    for i in range(len(zs[0])): # -1 because last point is not well computed

        kf.predict()
        fault_detector.review_measurement(zs[:,i],kf)
        # print('New measurement: \n{0}\n'.format(kf.x))
        kf.update(zs[:,i])
        # print('State space vector: \n{0}\n'.format(kf.x))
        xs.append(kf.x)

    xs = np.array(xs)

    # ==========================================================================
    # ========================== Comparison ====================================

    # Detection stats
    zipped_review = fault_detector.zipped_review()
    count = 0
    for res in zipped_review:
        if res[1]=="Failure":
            count += 1
    print('Wrong values: {0} out of {1} samples\n'.format(count,sample_nb))
    # pprint(zipped_review)

    t = np.linspace(0.,sample_nb,sample_nb)
    plt.plot(t,xs[:,0],'go',alpha = 0.1)
    plt.plot(t,pos,linestyle='dashed')
    plt.legend(['Compromised position','Healthy position'])
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.show()
