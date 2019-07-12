# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:29:27 2019

@author: qde
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from filterpy.common import kinematic_kf
from numpy.linalg import pinv, inv, det


def foo():
    return 1, 2, 3, 4, 5, 6


if __name__ == "__main__":
    a = list(foo())
    print(a)

    # H = np.array([[ 0.7,0,0, 0.1,0,0,-0.1,0,0],
    #               [-0.1,0,0, 0.2,0,0, 0.2,0,0],
    #               [ 0.6,0,0,-0.3,0,0, 0.6,0,0],
    #               [-0.4,0,0, 0.5,0,0,-0.2,0,0],
    #               [ 0.5,0,0, 0.4,0,0,-0.1,0,0],
    #               [-0.3,0,0,-0.2,0,0, 0.5,0,0]])
    # print("H=\n{0}\n".format(H))
    #
    # # dh = det(H)
    # # print("det(H)=\n{0}\n".format(dh))
    #
    # B = H@pinv(H.T@H)@H.T - np.eye(6)
    # print("B=\n{0}\n".format(B))
    #
    #
    # t = np.array([[0,0,0,-0.36,0,0.01]]).T
    # print("t=\n{0}\n".format(t))
    #
    # Bprim = B[:,3:]
    # print("B'=\n{0}\n".format(Bprim))
    #
    # d = np.array([[45,110,-130]]).T
    # print("d=\n{0}\n".format(d))
    #
    # aprim = pinv(Bprim)@(B@t) + (np.eye(3) - pinv(Bprim)@Bprim)@d
    # print("a'=\n{0}\n".format(aprim))
