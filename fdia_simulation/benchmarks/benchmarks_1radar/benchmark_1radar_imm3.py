# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 13:28:16 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman            import IMMEstimator
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterCV, RadarFilterCA, RadarFilterCT
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark



if __name__ == "__main__":
    trajectory = Track()
    states = trajectory.gen_cruise()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]
    radar = Radar(x=0,y=0)

    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 1.,   x0=x0,y0=y0,z0=z0,radar = radar)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 10.,  x0=x0,y0=y0,z0=z0,radar = radar)
    radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 20.,  x0=x0,y0=y0,z0=z0,radar = radar)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_ct]
    mu = [0.33, 0.33, 0.33]
    trans = np.array([[0.998, 0.001, 0.001],
                      [0.050, 0.900, 0.050],
                      [0.001, 0.001, 0.998]])
    imm = IMMEstimator(filters, mu, trans)

    benchmark_imm3 = Benchmark(radars = radar,radar_filter = imm,states = states)
    benchmark_imm3.launch_benchmark(with_nees = True)
