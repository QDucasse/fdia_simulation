# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 13:28:16 2019

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
from fdia_simulation.filters.radar_filter_ct import RadarFilterCT
from fdia_simulation.benchmarks.benchmark1radar import Benchmark1Radar



if __name__ == "__main__":
    trajectory = Track()
    states = trajectory.gen_takeoff()

    radar = Radar(x=0,y=2000)

    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 1.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 400.,x0 = 100.,y0=100.,radar = radar)
    radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 350.,x0 = 100.,y0=100.,radar = radar)
    filters = [radar_filter_cv, radar_filter_ca, radar_filter_ct]
    mu = [0.33, 0.33, 0.33]
    trans = np.array([[0.998, 0.001, 0.001],
                      [0.050, 0.900, 0.050],
                      [0.001, 0.001, 0.998]])
    imm = IMMEstimator(filters, mu, trans)

    benchmark_imm3 = Benchmark1Radar(radar = radar,radar_filter = imm,states = states)
    benchmark_imm3.launch_benchmark(with_nees = True)
