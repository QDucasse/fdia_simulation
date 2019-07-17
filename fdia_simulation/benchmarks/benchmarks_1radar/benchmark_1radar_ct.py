# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:29:18 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterCT
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark



if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()

    radar = Radar(x=0,y=2000)
    radar_filter_ct = RadarFilterCT(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)

    benchmark_ct = Benchmark(radar = radar,radar_filter = radar_filter_ct,states = states)
    benchmark_ct.launch_benchmark(with_nees = True)
