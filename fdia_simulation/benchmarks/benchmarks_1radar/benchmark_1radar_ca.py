# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:21 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models.radar                  import Radar
from fdia_simulation.attackers.mo_attacker         import MoAttacker
from fdia_simulation.filters.radar_filter_ca       import RadarFilterCA
from fdia_simulation.models.tracks                 import Track
from fdia_simulation.benchmarks.benchmark1radar    import Benchmark1Radar


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_landing()
    radar = Radar(x=2000,y=2000)
    radar_filter_ca = RadarFilterCA(dim_x = 9, dim_z = 3, q = 3600., x0=states[0,0],y0=states[3,0],z0=states[6,0], radar = radar)
    benchmark_ca = Benchmark1Radar(radar = radar, radar_filter = radar_filter_ca,states = states)
    benchmark_ca.launch_benchmark(with_nees = True)
