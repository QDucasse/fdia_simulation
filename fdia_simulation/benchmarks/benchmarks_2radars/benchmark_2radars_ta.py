# -*- coding: utf-8 -*-
"""
Created on Thu Jul 04 11:47:27 2019

@author: qde
"""
import numpy             as np
import matplotlib.pyplot as plt
from fdia_simulation.models     import Radar, Track
from fdia_simulation.filters    import MultipleRadarsFilter, RadarFilterTA
from fdia_simulation.attackers  import MoAttacker
from fdia_simulation.benchmarks import Benchmark


if __name__ == "__main__":
    #================== Position generation for the aircraft =====================
    trajectory = Track()
    states = trajectory.gen_takeoff()
    x0=states[0,0]
    y0=states[0,3]
    z0=states[0,6]

    # Radar 1: Precision radar
    radar1 = Radar(x = 500, y = 500)

    # Radar 2: Standard radar
    radar2 = Radar(x = 0, y = 1000,
                   r_noise_std = 5., theta_noise_std = 0.005, phi_noise_std = 0.005)

    radars = [radar1,radar2]
    radar_filter_ta = MultipleRadarsFilter(dim_x = 9, dim_z = 6, q = 400.,
                                           radars = radars, model = RadarFilterTA,
                                           x0 = x0, y0 = y0, z0 = z0)

    benchmark_ta = Benchmark(radars = radars,radar_filter = radar_filter_ta,states = states)
    benchmark_ta.launch_benchmark(with_nees = True)
