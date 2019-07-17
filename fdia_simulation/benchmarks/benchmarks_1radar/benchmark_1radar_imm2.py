# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:41:38 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from filterpy.kalman import IMMEstimator
from fdia_simulation.models.radar            import Radar
from fdia_simulation.models.tracks           import Track
from fdia_simulation.attackers.mo_attacker   import MoAttacker
from fdia_simulation.filters.radar_filter_cv import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca import RadarFilterCA
from fdia_simulation.benchmarks.benchmark1   import Benchmark



if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()

    radar = Radar(x=0,y=2000)

    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 1.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 400.,x0 = 100.,y0=100.,radar = radar)
    filters = [radar_filter_cv, radar_filter_ca]
    mu = [0.5, 0.5]
    trans = np.array([[0.998, 0.02],
                      [0.100, 0.900]])
    imm = IMMEstimator(filters, mu, trans)

    benchmark_imm2 = Benchmark(radar = radar,radar_filter = imm,states = states)
    benchmark_imm2.launch_benchmark(with_nees = True)
