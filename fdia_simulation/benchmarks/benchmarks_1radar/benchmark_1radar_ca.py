# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:21 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_cv       import RadarFilterCV
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.filters.radar_filter_ct       import RadarFilterCT
from fdia_simulation.filters.radar_filter_ta       import RadarFilterTA
from fdia_simulation.models.tracks                 import Track
from fdia_simulation.benchmarks.benchmark1radar    import Benchmark1Radar


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()
    xs, ys, zs = trajectory.output_positions(states)
    position_data = np.array(list(zip(xs,ys,zs)))
    radar = Radar(x=2000,y=2000)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    benchmark_ca = Benchmark1Radar(radar = radar,
                                   radar_filter = radar_filter_ca,
                                   pos_data = position_data,
                                   states = states)
    benchmark_ca.launch_benchmark(with_nees = True)

    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    benchmark_cv = Benchmark1Radar(radar = radar,
                                   radar_filter = radar_filter_cv,
                                   pos_data = position_data,
                                   states = states)
    benchmark_cv.launch_benchmark(with_nees = True)

    radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    benchmark_ct = Benchmark1Radar(radar = radar,
                                   radar_filter = radar_filter_ct,
                                   pos_data = position_data,
                                   states = states)
    benchmark_ct.launch_benchmark(with_nees = True)

    radar_filter_ta = RadarFilterTA(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)
    benchmark_ta = Benchmark1Radar(radar = radar,
                                   radar_filter = radar_filter_ta,
                                   pos_data = position_data,
                                   states = states)
    benchmark_ta.launch_benchmark(with_nees = True)
