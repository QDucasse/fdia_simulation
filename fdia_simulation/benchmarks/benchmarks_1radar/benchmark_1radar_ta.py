# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:22:16 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import RadarFilterTA
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()

    radar = Radar(x=0,y=2000)
    radar_filter_ta = RadarFilterTA(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)

    benchmark_ta = Benchmark(radars = radar,radar_filter = radar_filter_ta,states = states)
    benchmark_ta.launch_benchmark(with_nees = True)
