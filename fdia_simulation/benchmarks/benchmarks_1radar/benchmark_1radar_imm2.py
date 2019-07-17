# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:41:38 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterCV, RadarFilterCA
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    trajectory = Track()
    states = trajectory.gen_takeoff()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]

    radar = Radar(x=0,y=2000)

    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 1., x0=x0,y0=y0,z0=z0,radar = radar)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 400., x0=x0,y0=y0,z0=z0,radar = radar)
    filters = [radar_filter_cv, radar_filter_ca]
    mu = [0.5, 0.5]
    trans = np.array([[0.998, 0.02],
                      [0.100, 0.900]])
    imm = IMMEstimator(filters, mu, trans)

    benchmark_imm2 = Benchmark(radars = radar,radar_filter = imm,states = states)
    benchmark_imm2.launch_benchmark(with_nees = True)
