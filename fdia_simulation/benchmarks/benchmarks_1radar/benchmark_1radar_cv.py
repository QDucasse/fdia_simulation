# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:12 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterCV
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    trajectory = Track()
    states = trajectory.gen_takeoff()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]

    radar = Radar(x=0,y=2000)
    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 3600., x0=x0,y0=y0,z0=z0,radar = radar)

    benchmark_cv = Benchmark(radars = radar,radar_filter = radar_filter_cv,states = states)
    benchmark_cv.launch_benchmark(with_nees = True)
