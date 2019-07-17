# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:50:12 2019

@author: qde
"""

import numpy             as np
import matplotlib.pyplot as plt
from numpy.linalg   import inv
from fdia_simulation.models.radar            import Radar
from fdia_simulation.attackers.mo_attacker   import ExtendedMoAttacker
from fdia_simulation.filters.radar_filter_cv import RadarFilterCV
from fdia_simulation.models.tracks           import Track
from fdia_simulation.benchmarks.benchmark    import Benchmark


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()

    radar = Radar(x=0,y=2000)
    radar_filter_cv = RadarFilterCV(dim_x = 9, dim_z = 3, q = 3600., x0=100.,y0=100., radar = radar)

    benchmark_cv = Benchmark(radar = radar,radar_filter = radar_filter_cv,states = states)
    benchmark_cv.launch_benchmark(with_nees = True)
